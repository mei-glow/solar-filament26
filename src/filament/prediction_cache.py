from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .experiment import sha256_file, write_json
from .postprocess import mask_to_rle_object, rle_from_json, rle_to_json


class PredictionCache:
    def __init__(self, root: Path, checkpoint: Path, settings: dict, reuse: bool = True):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = root / "metadata.json"
        metadata = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            **settings,
        }
        if self.metadata_path.is_file():
            existing = json.loads(self.metadata_path.read_text())
            if existing != metadata:
                if reuse:
                    raise RuntimeError(
                        f"Prediction cache metadata mismatch at {root}. Use a new --run-name or --no-reuse-cache."
                    )
                raise RuntimeError(f"Refusing to overwrite incompatible cache: {root}")
        else:
            write_json(self.metadata_path, metadata)

    def path_for(self, file_name: str) -> Path:
        return self.root / f"{Path(file_name).stem}.json.gz"

    def has(self, file_name: str) -> bool:
        return self.path_for(file_name).is_file()

    def save(self, file_name: str, masks: np.ndarray, scores: np.ndarray, boxes: np.ndarray) -> None:
        payload = {
            "file_name": file_name,
            "instances": [
                {"rle": rle_to_json(mask_to_rle_object(mask)), "score": float(score), "box": box.tolist()}
                for mask, score, box in zip(masks, scores, boxes)
            ],
        }
        target = self.path_for(file_name)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        temporary.replace(target)

    def load(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with gzip.open(self.path_for(file_name), "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        instances = payload["instances"]
        if not instances:
            return (
                np.empty((0, 1, 1), dtype=bool),
                np.empty(0, dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )
        from pycocotools import mask as mask_utils

        masks = np.stack([mask_utils.decode(rle_from_json(item["rle"])).astype(bool) for item in instances])
        scores = np.asarray([item["score"] for item in instances], dtype=np.float32)
        boxes = np.asarray([item["box"] for item in instances], dtype=np.float32)
        return masks, scores, boxes
