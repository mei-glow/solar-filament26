from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .experiment import sha256_file, write_json
from .semantic_data import build_features, radial_map
from .semantic_model import build_semantic_model


def load_semantic_checkpoint(checkpoint_path: Path, device: torch.device):
    # These are checkpoints produced by train_semantic.py and include the
    # experiment config (including Path values), not just tensor weights.
    # PyTorch 2.6 defaults weights_only=True, which rejects that trusted
    # metadata, so load the complete locally-created payload explicitly.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    encoder_weights = None
    model = build_semantic_model(config["architecture"], config["encoder"], encoder_weights)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, config


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


@torch.no_grad()
def _predict_batch(model, batch: torch.Tensor, tta: bool) -> torch.Tensor:
    output = model(batch).sigmoid()
    if not tta:
        return output
    horizontal = torch.flip(model(torch.flip(batch, dims=(3,))).sigmoid(), dims=(3,))
    vertical = torch.flip(model(torch.flip(batch, dims=(2,))).sigmoid(), dims=(2,))
    return (output + horizontal + vertical) / 3.0


@torch.no_grad()
def infer_semantic_image(
    model,
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: float,
    feature_mode: str,
    batch_size: int = 4,
    tta: bool = False,
) -> np.ndarray:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    height, width = image.shape
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    ys, xs = _positions(height, tile_size, stride), _positions(width, tile_size, stride)
    radial = radial_map(height, width) if feature_mode == "radial" else None
    hann = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
    weight = np.clip(hann, 0.05, None)
    output_sum = np.zeros((3, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    tiles, locations = [], []

    def flush() -> None:
        if not tiles:
            return
        tensor = torch.from_numpy(np.stack(tiles)).to(device, non_blocking=True)
        prediction = _predict_batch(model, tensor, tta).cpu().numpy()
        for values, (y, x, actual_h, actual_w) in zip(prediction, locations):
            local_weight = weight[:actual_h, :actual_w]
            output_sum[:, y : y + actual_h, x : x + actual_w] += values[:, :actual_h, :actual_w] * local_weight
            weight_sum[y : y + actual_h, x : x + actual_w] += local_weight
        tiles.clear()
        locations.clear()

    for y in ys:
        for x in xs:
            image_tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            actual_h, actual_w = image_tile.shape
            if actual_h != tile_size or actual_w != tile_size:
                image_tile = cv2.copyMakeBorder(
                    image_tile, 0, tile_size - actual_h, 0, tile_size - actual_w, cv2.BORDER_REFLECT_101
                )
            radial_tile = None
            if radial is not None:
                radial_tile = radial[y : min(y + tile_size, height), x : min(x + tile_size, width)]
                if radial_tile.shape != (tile_size, tile_size):
                    radial_tile = cv2.copyMakeBorder(
                        radial_tile, 0, tile_size - actual_h, 0, tile_size - actual_w, cv2.BORDER_REFLECT_101
                    )
            tiles.append(build_features(image_tile, feature_mode, radial_tile))
            locations.append((y, x, actual_h, actual_w))
            if len(tiles) >= batch_size:
                flush()
    flush()
    return output_sum / np.maximum(weight_sum[None], 1e-6)


class SemanticPredictionCache:
    def __init__(self, root: Path, checkpoint: Path, settings: dict):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = root / "metadata.json"
        metadata = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint), **settings}
        if self.metadata_path.is_file():
            existing = json.loads(self.metadata_path.read_text())
            if existing != metadata:
                raise RuntimeError(f"Semantic cache mismatch: {root}. Use a different semantic run name.")
        else:
            write_json(self.metadata_path, metadata)

    def path_for(self, file_name: str) -> Path:
        return self.root / f"{Path(file_name).stem}.npz"

    def has(self, file_name: str) -> bool:
        return self.path_for(file_name).is_file()

    def save(self, file_name: str, probability: np.ndarray) -> None:
        target = self.path_for(file_name)
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, probability=probability.astype(np.float16))
        temporary.replace(target)

    def load(self, file_name: str) -> np.ndarray:
        with np.load(self.path_for(file_name)) as payload:
            return payload["probability"].astype(np.float32)
