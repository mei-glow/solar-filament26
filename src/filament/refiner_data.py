from __future__ import annotations

import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data.annotators import select_annotation_records


@lru_cache(maxsize=8)
def _read_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _polygon_mask(shape: tuple[int, int], polygons: list[list[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def _square_crop(array: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    half = side // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + side, y0 + side
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - array.shape[1]), max(0, y1 - array.shape[0])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(array.shape[1], x1), min(array.shape[0], y1)
    crop = array[y0:y1, x0:x1]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    return crop


def _normalize(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    lo, hi = np.percentile(values, (1, 99))
    return np.clip((values - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)


def _coarse_seed(target: np.ndarray, training: bool) -> np.ndarray:
    height, width = target.shape
    small = cv2.resize(target, (max(8, width // 8), max(8, height // 8)), interpolation=cv2.INTER_AREA)
    seed = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    seed = (seed > (random.uniform(0.25, 0.55) if training else 0.4)).astype(np.uint8)
    if training:
        kernel = np.ones((random.choice((3, 5, 7)),) * 2, np.uint8)
        seed = cv2.erode(seed, kernel) if random.random() < 0.6 else cv2.dilate(seed, kernel)
    return seed.astype(np.float32)


class RefinerDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        image_dir: Path,
        manifest_path: Path,
        split: str,
        crop_size: int = 512,
        annotation_strategy: str = "medoid",
        seed: int = 3020,
        negative_fraction: float = 0.25,
        training: bool = True,
    ):
        with annotation_path.open() as handle:
            self.coco = json.load(handle)
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.training = training
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["split"] == split].copy()
        manifest = select_annotation_records(manifest, self.coco, annotation_strategy, seed)
        selected_ids = set(manifest["image_id"].astype(str))
        self.images = {str(image["id"]): image for image in self.coco["images"] if str(image["id"]) in selected_ids}
        annotations: dict[str, list[dict]] = defaultdict(list)
        for annotation in self.coco["annotations"]:
            image_id = str(annotation["image_id"])
            if image_id in selected_ids:
                annotations[image_id].append(annotation)
        self.annotations = annotations
        self.positives = [
            (image_id, annotation)
            for image_id, instances in annotations.items()
            for annotation in instances
        ]
        self.image_ids = sorted(self.images)
        negative_count = int(len(self.positives) * negative_fraction / max(1.0 - negative_fraction, 1e-6))
        self.items = [("positive", item) for item in self.positives] + [
            ("negative", self.image_ids[index % len(self.image_ids)]) for index in range(negative_count)
        ]
        random.Random(seed + (0 if split == "train" else 10000)).shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        kind, payload = self.items[index]
        if kind == "positive":
            image_id, annotation = payload
        else:
            image_id, annotation = payload, None
        info = self.images[str(image_id)]
        image = _read_gray(str(self.image_dir / info["file_name"]))

        if annotation is not None:
            target_full = _polygon_mask(image.shape, annotation["segmentation"])
            x, y, width, height = map(float, annotation["bbox"])
            object_side = max(width, height)
            side = max(self.crop_size, int(np.ceil(object_side * random.uniform(1.3, 1.8))))
            jitter = 0.10 * side if self.training else 0.0
            cx = x + width / 2 + random.uniform(-jitter, jitter)
            cy = y + height / 2 + random.uniform(-jitter, jitter)
        else:
            target_full = np.zeros_like(image, dtype=np.uint8)
            side = self.crop_size
            exclusion = np.zeros_like(image, dtype=np.uint8)
            for instance in self.annotations[str(image_id)]:
                exclusion |= _polygon_mask(image.shape, instance["segmentation"])
            # Pick a dark on-disk crop that does not contain a labeled filament.
            rng = np.random.default_rng(index + 7919)
            xs = rng.integers(side // 2, max(side // 2 + 1, image.shape[1] - side // 2), size=256)
            ys = rng.integers(side // 2, max(side // 2 + 1, image.shape[0] - side // 2), size=256)
            disk_x, disk_y = image.shape[1] / 2, image.shape[0] / 2
            radius = 0.43 * min(image.shape)
            on_disk = (xs - disk_x) ** 2 + (ys - disk_y) ** 2 < radius**2
            candidates = []
            for candidate, (x_value, y_value) in enumerate(zip(xs, ys)):
                if not on_disk[candidate]:
                    continue
                x0, x1 = int(x_value - side // 2), int(x_value + side // 2)
                y0, y1 = int(y_value - side // 2), int(y_value + side // 2)
                if exclusion[y0:y1, x0:x1].sum() == 0:
                    candidates.append(candidate)
            if candidates:
                chosen = min(candidates, key=lambda candidate: int(image[ys[candidate], xs[candidate]]))
            else:
                on_disk_indices = np.flatnonzero(on_disk)
                chosen = int(on_disk_indices[np.argmin(image[ys[on_disk_indices], xs[on_disk_indices]])])
            cx, cy = float(xs[chosen]), float(ys[chosen])

        image_crop = _square_crop(image, cx, cy, side)
        target = _square_crop(target_full, cx, cy, side)
        if side != self.crop_size:
            image_crop = cv2.resize(image_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        target = (target > 0).astype(np.float32)

        if self.training:
            rotations = random.randrange(4)
            image_crop, target = np.rot90(image_crop, rotations).copy(), np.rot90(target, rotations).copy()
            if random.random() < 0.5:
                image_crop, target = np.fliplr(image_crop).copy(), np.fliplr(target).copy()
            if random.random() < 0.5:
                image_crop, target = np.flipud(image_crop).copy(), np.flipud(target).copy()

        normalized = _normalize(image_crop)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_crop)
        clahe = _normalize(clahe)
        seed_mask = _coarse_seed(target, self.training) if annotation is not None else np.zeros_like(target)
        features = np.stack([normalized, clahe, seed_mask], axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(features),
            "target": torch.from_numpy(target[None]),
            "file_name": info["file_name"],
        }
