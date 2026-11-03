from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.filters import frangi
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from .data.annotators import select_annotation_records


@lru_cache(maxsize=8)
def read_gray(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def polygons_to_union(shape: tuple[int, int], annotations: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for annotation in annotations:
        for polygon in annotation["segmentation"]:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 1)
    return mask


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    lo, hi = np.percentile(values, (1, 99))
    return np.clip((values - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)


@lru_cache(maxsize=4)
def radial_map(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt(((xx - (width - 1) / 2) / (width / 2)) ** 2 + ((yy - (height - 1) / 2) / (height / 2)) ** 2)
    return np.clip(radius, 0.0, 1.0).astype(np.float32)


def build_features(image: np.ndarray, feature_mode: str, radial: np.ndarray | None = None) -> np.ndarray:
    normalized = normalize_intensity(image)
    clahe_image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    local_contrast = normalize_intensity(clahe_image)
    if feature_mode == "radial":
        if radial is None:
            radial = radial_map(*image.shape)
        third = radial.astype(np.float32)
    elif feature_mode == "frangi":
        third = frangi(normalized, sigmas=(1, 2, 4), black_ridges=True).astype(np.float32)
        maximum = float(third.max())
        if maximum > 0:
            third /= maximum
    elif feature_mode == "intensity":
        third = normalized
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    return np.stack([normalized, local_contrast, third], axis=0).astype(np.float32)


def crop_at(array: np.ndarray, cx: int, cy: int, size: int, *, reflect: bool) -> np.ndarray:
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - array.shape[1]), max(0, y1 - array.shape[0])
    crop = array[max(0, y0) : min(array.shape[0], y1), max(0, x0) : min(array.shape[1], x1)]
    if any((left, top, right, bottom)):
        border = cv2.BORDER_REFLECT_101 if reflect else cv2.BORDER_CONSTANT
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, border, value=0)
    return crop


class SemanticTileDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path,
        image_dir: Path,
        manifest_path: Path,
        split: str,
        tile_size: int,
        samples: int,
        annotation_strategy: str,
        feature_mode: str,
        positive_fraction: float,
        seed: int,
        training: bool,
    ):
        with annotation_path.open() as handle:
            coco = json.load(handle)
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["split"] == split].copy()
        manifest = select_annotation_records(manifest, coco, annotation_strategy, seed)
        selected = set(manifest["image_id"].astype(str))
        self.images = {str(image["id"]): image for image in coco["images"] if str(image["id"]) in selected}
        self.annotations: dict[str, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            image_id = str(annotation["image_id"])
            if image_id in selected:
                self.annotations[image_id].append(annotation)
        self.image_ids = sorted(self.images)
        self.image_dir = image_dir
        self.tile_size = tile_size
        self.samples = samples
        self.feature_mode = feature_mode
        self.positive_fraction = positive_fraction
        self.seed = seed
        self.training = training
        self._mask_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples

    def _mask(self, image_id: str, shape: tuple[int, int]) -> np.ndarray:
        if image_id not in self._mask_cache:
            if len(self._mask_cache) >= 8:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[image_id] = polygons_to_union(shape, self.annotations[image_id])
        return self._mask_cache[image_id]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(None if self.training else self.seed + index * 104729)
        image_id = self.image_ids[int(rng.integers(0, len(self.image_ids)))]
        info = self.images[image_id]
        image = read_gray(str(self.image_dir / info["file_name"]))
        target_full = self._mask(image_id, image.shape)
        use_positive = bool(target_full.any()) and rng.random() < self.positive_fraction
        if use_positive:
            ys, xs = np.where(target_full)
            chosen = int(rng.integers(0, len(xs)))
            cx, cy = int(xs[chosen]), int(ys[chosen])
            jitter = self.tile_size // 6
            cx += int(rng.integers(-jitter, jitter + 1))
            cy += int(rng.integers(-jitter, jitter + 1))
        else:
            cx = int(rng.integers(self.tile_size // 2, max(self.tile_size // 2 + 1, image.shape[1] - self.tile_size // 2)))
            cy = int(rng.integers(self.tile_size // 2, max(self.tile_size // 2 + 1, image.shape[0] - self.tile_size // 2)))
        cx = int(np.clip(cx, 0, image.shape[1] - 1))
        cy = int(np.clip(cy, 0, image.shape[0] - 1))

        image_crop = crop_at(image, cx, cy, self.tile_size, reflect=True)
        mask = crop_at(target_full, cx, cy, self.tile_size, reflect=False).astype(np.uint8)
        full_radial = radial_map(*image.shape) if self.feature_mode == "radial" else None
        radial_crop = crop_at(full_radial, cx, cy, self.tile_size, reflect=True) if full_radial is not None else None
        features = build_features(image_crop, self.feature_mode, radial_crop)

        if self.training:
            rotations = int(rng.integers(0, 4))
            features = np.rot90(features, rotations, axes=(1, 2)).copy()
            mask = np.rot90(mask, rotations).copy()
            if rng.random() < 0.5:
                features, mask = features[:, :, ::-1].copy(), mask[:, ::-1].copy()
            if rng.random() < 0.5:
                features, mask = features[:, ::-1, :].copy(), mask[::-1, :].copy()
            gain = float(rng.uniform(0.90, 1.10))
            features[:2] = np.clip(features[:2] * gain, 0.0, 1.0)

        centerline = skeletonize(mask > 0).astype(np.float32)
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel).astype(np.float32)
        targets = np.stack([mask.astype(np.float32), centerline, boundary], axis=0)
        return {
            "image": torch.from_numpy(features),
            "target": torch.from_numpy(targets),
            "file_name": info["file_name"],
        }
