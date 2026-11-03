from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _safe_qcut(values: pd.Series, q: int = 4) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=min(q, len(values)), labels=False, duplicates="drop").fillna(0).astype(int)


def build_fold_manifest(annotation_path: Path, output_path: Path, folds: int = 5, seed: int = 3020) -> pd.DataFrame:
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = pd.DataFrame(coco["images"])
    anns_by_id: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_id[str(annotation["image_id"])].append(annotation)

    image_stats = []
    for image in coco["images"]:
        anns = anns_by_id[str(image["id"])]
        image_stats.append(
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "instance_count": len(anns),
                "foreground_area": float(sum(float(a.get("area", 0.0)) for a in anns)),
            }
        )
    stats = pd.DataFrame(image_stats)
    physical = stats.groupby("file_name", as_index=False).agg(
        instance_count=("instance_count", "mean"),
        foreground_area=("foreground_area", "mean"),
        annotator_count=("image_id", "size"),
    )
    physical["source"] = physical["file_name"].str.replace(r"\.[^.]+$", "", regex=True).str[-2:]
    physical["year_band"] = (physical["file_name"].str[:4].astype(int) // 4).astype(str)
    physical["count_bin"] = _safe_qcut(physical["instance_count"])
    physical["stratum"] = physical["source"] + "_" + physical["count_bin"].astype(str)
    counts = physical["stratum"].value_counts()
    rare = counts[counts < folds].index
    physical.loc[physical["stratum"].isin(rare), "stratum"] = physical.loc[
        physical["stratum"].isin(rare), "source"
    ]
    counts = physical["stratum"].value_counts()
    physical.loc[physical["stratum"].map(counts) < folds, "stratum"] = "other"

    physical["fold"] = -1
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, val_indices) in enumerate(splitter.split(physical, physical["stratum"])):
        physical.loc[val_indices, "fold"] = fold
    if (physical["fold"] < 0).any():
        raise RuntimeError("Some physical images were not assigned to a fold")

    manifest = images.rename(columns={"id": "image_id"}).merge(
        physical[["file_name", "fold", "source", "year_band", "instance_count", "foreground_area", "annotator_count"]],
        on="file_name",
        how="left",
        validate="many_to_one",
    )
    manifest["sample_stem"] = manifest["image_id"].astype(str).str.replace(r"[/\\]", "_", regex=True)
    manifest["annotator"] = manifest["image_id"].astype(str).str.split("-", n=1).str[0]
    manifest = manifest[
        ["sample_stem", "image_id", "file_name", "fold", "annotator", "source", "year_band", "instance_count", "foreground_area", "annotator_count"]
    ].sort_values(["file_name", "image_id"])

    leakage = manifest.groupby("file_name")["fold"].nunique().max()
    if leakage != 1:
        raise RuntimeError("Physical-image leakage detected in fold manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    metadata = {
        "folds": folds,
        "seed": seed,
        "physical_images": int(manifest["file_name"].nunique()),
        "annotation_records": len(manifest),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest


def load_or_build_manifest(annotation_path: Path, manifest_path: Path, folds: int, seed: int, rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not manifest_path.is_file():
        return build_fold_manifest(annotation_path, manifest_path, folds, seed)
    manifest = pd.read_csv(manifest_path)
    if manifest.groupby("file_name")["fold"].nunique().max() != 1:
        raise RuntimeError(f"Physical-image leakage in {manifest_path}")
    return manifest
