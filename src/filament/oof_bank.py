from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from .postprocess import mask_to_rle_object, rle_from_json, rle_to_json


class OOFProposalBank:
    """Portable, fold-aware storage for low-confidence detector proposals."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, file_name: str) -> Path:
        return self.root / f"{Path(file_name).stem}.json.gz"

    def files(self) -> list[Path]:
        return sorted(self.root.glob("*.json.gz"))

    def save(self, file_name: str, fold: int, instances: list[dict], metadata: dict | None = None) -> None:
        payload = {"file_name": file_name, "fold": int(fold), "instances": instances, "metadata": metadata or {}}
        target = self.path_for(file_name)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        temporary.replace(target)

    def load(self, file_name_or_path: str | Path, decode_masks: bool = True) -> dict:
        path = Path(file_name_or_path)
        if not path.is_file():
            path = self.path_for(str(file_name_or_path))
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if decode_masks:
            from pycocotools import mask as mask_utils

            for instance in payload["instances"]:
                instance["mask"] = mask_utils.decode(rle_from_json(instance["rle"])).astype(bool)
        return payload


def proposal_record(mask: np.ndarray, score: float, box: np.ndarray, source: str) -> dict:
    return {
        "rle": rle_to_json(mask_to_rle_object(mask)),
        "score": float(score),
        "box": np.asarray(box, dtype=float).tolist(),
        "source": source,
    }
