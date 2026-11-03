from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from .refiner_model import build_refiner
from .semantic_data import crop_at, normalize_intensity
from .sira_data import proposal_centers, seed_affinity
from .postprocess import rle_from_json


def load_sira(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_refiner(config["architecture"], config["encoder"], None, int(config["in_channels"]))
    model.load_state_dict(payload["model"]); model.to(device).eval()
    return model, config


def _paste(sum_map: np.ndarray, weight_map: np.ndarray, patch: np.ndarray, cx: int, cy: int) -> None:
    size = patch.shape[0]; half = size // 2
    x0, y0 = cx - half, cy - half; x1, y1 = x0 + size, y0 + size
    sx0, sy0 = max(0, -x0), max(0, -y0); sx1, sy1 = size - max(0, x1 - sum_map.shape[1]), size - max(0, y1 - sum_map.shape[0])
    dx0, dy0 = max(0, x0), max(0, y0); dx1, dy1 = min(sum_map.shape[1], x1), min(sum_map.shape[0], y1)
    weight = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    weight = np.clip(weight, .05, None)[sy0:sy1, sx0:sx1]
    sum_map[dy0:dy1, dx0:dx1] += patch[sy0:sy1, sx0:sx1] * weight
    weight_map[dy0:dy1, dx0:dx1] += weight


@torch.no_grad()
def refine_candidates(model, config: dict, image: np.ndarray, candidates: list[dict], semantic: np.ndarray,
                      device: torch.device, batch_size: int = 4) -> list[dict]:
    crop_size = int(config["crop_size"]); neighbor_channel = bool(config.get("neighbor_channel", False))
    clahe_full = cv2.createCLAHE(2.0, (8, 8)).apply(image)
    outputs = []
    competition_count = None
    if neighbor_channel:
        competition_count = np.zeros(image.shape, np.uint16)
        for item in candidates:
            current = item.get("mask")
            if current is None: current = mask_utils.decode(rle_from_json(item["rle"])).astype(bool)
            competition_count += current.astype(np.uint16)
    for candidate_index, candidate in enumerate(candidates):
        current_mask = candidate.get("mask")
        if current_mask is None: current_mask = mask_utils.decode(rle_from_json(candidate["rle"])).astype(bool)
        seed = current_mask.astype(np.uint8)
        neighbor = np.zeros_like(seed)
        if neighbor_channel:
            neighbor = (competition_count - seed.astype(np.uint16) > 0).astype(np.uint8)
        probability_sum = np.zeros(image.shape, np.float32); weight_sum = np.zeros(image.shape, np.float32)
        tensors, locations = [], []
        for cx, cy in proposal_centers(candidate["box"], crop_size):
            gray = crop_at(image, cx, cy, crop_size, reflect=True)
            clahe = crop_at(clahe_full, cx, cy, crop_size, reflect=True)
            seed_crop = crop_at(seed, cx, cy, crop_size, reflect=False)
            sem_crop = crop_at(semantic, cx, cy, crop_size, reflect=False)
            features = [normalize_intensity(gray), normalize_intensity(clahe), seed_crop.astype(np.float32), seed_affinity(seed_crop), sem_crop]
            if neighbor_channel: features.append(crop_at(neighbor, cx, cy, crop_size, reflect=False).astype(np.float32))
            tensors.append(np.stack(features)); locations.append((cx, cy))
            if len(tensors) >= batch_size:
                prediction = model(torch.from_numpy(np.stack(tensors)).to(device)).sigmoid().cpu().numpy()[:, 0]
                for patch, (px, py) in zip(prediction, locations): _paste(probability_sum, weight_sum, patch, px, py)
                tensors.clear(); locations.clear()
        if tensors:
            prediction = model(torch.from_numpy(np.stack(tensors)).to(device)).sigmoid().cpu().numpy()[:, 0]
            for patch, (px, py) in zip(prediction, locations): _paste(probability_sum, weight_sum, patch, px, py)
        probability = probability_sum / np.maximum(weight_sum, 1e-6)
        active = weight_sum > 0
        values = probability[active]
        ys, xs = np.where(active)
        if len(xs):
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        else:
            x0 = y0 = 0; x1 = y1 = 1
        outputs.append({**candidate, "probability": probability[y0:y1, x0:x1], "bounds": (x0, y0, x1, y1),
                        "mean_probability": float(values.mean()) if len(values) else 0.0,
                        "q90_probability": float(np.quantile(values, .9)) if len(values) else 0.0,
                        "seed_agreement": float(probability[seed > 0].mean()) if seed.any() else 0.0})
    return outputs
