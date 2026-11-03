from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from .data.annotators import select_annotation_records
from .oof_bank import OOFProposalBank
from .postprocess import polygon_to_mask, polygon_to_rle, rle_from_json
from .semantic_data import crop_at, normalize_intensity


@lru_cache(maxsize=8)
def _gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


@lru_cache(maxsize=3)
def _bank_payload(path: str) -> dict:
    file_path = Path(path)
    return OOFProposalBank(file_path.parent).load(file_path, decode_masks=False)


@lru_cache(maxsize=2)
def _semantic_probability(path: str) -> np.ndarray:
    with np.load(path) as payload:
        values = payload["probability"].astype(np.float32)
    return values[0] if values.ndim == 3 else values


def seed_affinity(seed: np.ndarray) -> np.ndarray:
    seed_u8 = seed.astype(np.uint8)
    inside = cv2.distanceTransform(seed_u8, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - seed_u8, cv2.DIST_L2, 5)
    signed = inside - outside
    scale = max(float(np.percentile(np.abs(signed), 95)), 1.0)
    return np.clip(0.5 + 0.5 * signed / scale, 0.0, 1.0).astype(np.float32)


def proposal_centers(box: list[float], crop_size: int) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = map(float, box)
    step = max(64, int(crop_size * 0.65))
    xs = [int(round((x0 + x1) / 2))] if x1 - x0 <= crop_size * .8 else list(range(int(x0 + crop_size / 2), int(x1), step))
    ys = [int(round((y0 + y1) / 2))] if y1 - y0 <= crop_size * .8 else list(range(int(y0 + crop_size / 2), int(y1), step))
    if not xs: xs = [int(round((x0 + x1) / 2))]
    if not ys: ys = [int(round((y0 + y1) / 2))]
    return [(x, y) for y in ys for x in xs]


class SIRADataset(Dataset):
    """Actual OOF proposal crops. No target-derived/synthetic seeds are used."""

    def __init__(
        self, annotation_path: Path, image_dir: Path, manifest_path: Path,
        bank_root: Path, split_folds: set[int], crop_size: int = 512,
        annotation_strategy: str = "medoid", semantic_root: Path | None = None,
        neighbor_channel: bool = False, min_proposal_iou: float = 0.02,
        seed: int = 3020, training: bool = True, negative_fraction: float = 0.35,
    ):
        with annotation_path.open() as handle: self.coco = json.load(handle)
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["fold"].isin(split_folds)].copy()
        manifest = select_annotation_records(manifest, self.coco, annotation_strategy, seed)
        self.records = {row.file_name: str(row.image_id) for row in manifest.itertuples(index=False)}
        self.images = {str(item["id"]): item for item in self.coco["images"]}
        anns: dict[str, list[dict]] = defaultdict(list)
        for item in self.coco["annotations"]: anns[str(item["image_id"])].append(item)
        self.annotations = anns; self.image_dir = Path(image_dir); self.bank_root = Path(bank_root)
        self.semantic_root = Path(semantic_root) if semantic_root else None
        self.crop_size = crop_size; self.neighbor_channel = neighbor_channel; self.training = training
        self._target_cache: dict[str, list[np.ndarray]] = {}
        self.bank_paths: dict[str, Path] = {}
        for fold in split_folds:
            for path in (self.bank_root / f"fold{fold}").glob("*.json.gz"):
                self.bank_paths[path.name.removesuffix(".json.gz")] = path
        # Match filenames robustly because extension may differ.
        stems = self.bank_paths
        self.items, self.file_folds = [], {}
        for file_name, image_id in sorted(self.records.items()):
            bank_path = stems.get(Path(file_name).stem)
            if bank_path is None: raise FileNotFoundError(f"OOF bank missing {file_name} under {self.bank_root}")
            payload = OOFProposalBank(bank_path.parent).load(bank_path, decode_masks=False)
            self.file_folds[file_name] = int(payload["fold"])
            info = self.images[image_id]
            pred_rles = [rle_from_json(item["rle"]) for item in payload["instances"]]
            gt_rles = [polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in anns[image_id]]
            ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)) if pred_rles and gt_rles else np.zeros((len(pred_rles), len(gt_rles)))
            for candidate_index, instance in enumerate(payload["instances"]):
                best_gt = int(np.argmax(ious[candidate_index])) if ious.shape[1] else -1
                best_iou = float(ious[candidate_index, best_gt]) if best_gt >= 0 else 0.0
                # Keep all hard negatives and every proposal with even weak GT evidence.
                matched = best_gt if best_iou >= min_proposal_iou else -1
                for cx, cy in proposal_centers(instance["box"], crop_size):
                    self.items.append((file_name, candidate_index, cx, cy, matched, best_iou))
        if training and 0 <= negative_fraction < 1:
            positive = [item for item in self.items if item[4] >= 0]
            negative = [item for item in self.items if item[4] < 0]
            maximum = int(len(positive) * negative_fraction / max(1.0 - negative_fraction, 1e-6))
            rng = np.random.default_rng(seed)
            if len(negative) > maximum:
                chosen = rng.choice(len(negative), maximum, replace=False)
                negative = [negative[int(index)] for index in chosen]
            self.items = positive + negative
            rng.shuffle(self.items)

    def __len__(self) -> int: return len(self.items)

    def _targets(self, file_name: str) -> list[np.ndarray]:
        if file_name in self._target_cache:
            return self._target_cache[file_name]
        image_id = self.records[file_name]; info = self.images[image_id]
        targets = [polygon_to_mask(item["segmentation"], info["height"], info["width"]) for item in self.annotations[image_id]]
        if len(self._target_cache) >= 4:
            self._target_cache.pop(next(iter(self._target_cache)))
        self._target_cache[file_name] = targets
        return targets

    def _semantic(self, file_name: str, shape: tuple[int, int]) -> np.ndarray:
        if self.semantic_root is None: return np.zeros(shape, dtype=np.float32)
        fold = self.file_folds[file_name]
        path = self.semantic_root / f"fold{fold}" / f"{Path(file_name).stem}.npz"
        if not path.is_file(): raise FileNotFoundError(f"OOF semantic cache missing: {path}")
        return _semantic_probability(str(path))

    def __getitem__(self, index: int) -> dict:
        file_name, candidate_index, cx, cy, matched, match_iou = self.items[index]
        bank_path = self.bank_paths[Path(file_name).stem]
        payload = _bank_payload(str(bank_path)); proposal = payload["instances"][candidate_index]
        image = _gray(str(self.image_dir / file_name)); seed = mask_utils.decode(rle_from_json(proposal["rle"])).astype(np.uint8)
        targets = self._targets(file_name)
        target = targets[matched].astype(np.uint8) if matched >= 0 else np.zeros_like(seed)
        exclusion = np.zeros_like(seed)
        if matched >= 0:
            for gt_index, gt in enumerate(targets):
                if gt_index != matched: exclusion |= gt.astype(np.uint8)
        neighbor = np.zeros_like(seed)
        if self.neighbor_channel:
            half = self.crop_size // 2
            crop_x0, crop_y0, crop_x1, crop_y1 = cx - half, cy - half, cx + half, cy + half
            for other_index, other in enumerate(payload["instances"]):
                if other_index == candidate_index:
                    continue
                x0, y0, x1, y1 = other["box"]
                if x1 < crop_x0 or x0 > crop_x1 or y1 < crop_y0 or y0 > crop_y1:
                    continue
                neighbor |= mask_utils.decode(rle_from_json(other["rle"])).astype(np.uint8)
        gray = crop_at(image, cx, cy, self.crop_size, reflect=True)
        seed_crop = crop_at(seed, cx, cy, self.crop_size, reflect=False)
        target_crop = crop_at(target, cx, cy, self.crop_size, reflect=False)
        exclusion_crop = crop_at(exclusion, cx, cy, self.crop_size, reflect=False)
        neighbor_crop = crop_at(neighbor, cx, cy, self.crop_size, reflect=False)
        semantic_crop = crop_at(self._semantic(file_name, image.shape), cx, cy, self.crop_size, reflect=False)
        clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        features = [normalize_intensity(gray), normalize_intensity(clahe), seed_crop.astype(np.float32), seed_affinity(seed_crop), semantic_crop]
        if self.neighbor_channel: features.append(neighbor_crop.astype(np.float32))
        features = np.stack(features).astype(np.float32)
        if self.training:
            rng = np.random.default_rng()
            rotation = int(rng.integers(0, 4)); features = np.rot90(features, rotation, axes=(1, 2)).copy()
            target_crop = np.rot90(target_crop, rotation).copy(); exclusion_crop = np.rot90(exclusion_crop, rotation).copy()
            if rng.random() < .5:
                features = features[:, :, ::-1].copy(); target_crop = target_crop[:, ::-1].copy(); exclusion_crop = exclusion_crop[:, ::-1].copy()
            if rng.random() < .5:
                features = features[:, ::-1, :].copy(); target_crop = target_crop[::-1, :].copy(); exclusion_crop = exclusion_crop[::-1, :].copy()
        weight = 1.0 if matched >= 0 else 0.5
        if 0 < match_iou < .1: weight = .25
        return {"image": torch.from_numpy(features), "target": torch.from_numpy(target_crop[None].astype(np.float32)),
                "exclusion": torch.from_numpy(exclusion_crop[None].astype(np.float32)), "weight": torch.tensor(weight, dtype=torch.float32),
                "file_name": file_name, "candidate_index": candidate_index, "cx": cx, "cy": cy, "match_iou": match_iou}
