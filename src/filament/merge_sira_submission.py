from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import sha256_file, write_json
from .infer_sira_test_fold import FEATURES, _test_images
from .oof_bank import OOFProposalBank
from .postprocess import mask_to_rle, resolve_instances_scored, rle_from_json


def _cluster_candidates(candidates: list[dict], cluster_iou: float, min_fold_support: int) -> list[dict]:
    if not candidates:
        return []
    candidates.sort(key=lambda item: item["quality_score"], reverse=True)
    rles = [item["rle"] for item in candidates]
    pairwise = mask_utils.iou(rles, rles, [0] * len(rles))
    clusters: list[dict] = []
    for index, candidate in enumerate(candidates):
        best_cluster, best_iou = None, 0.0
        for cluster in clusters:
            current = float(pairwise[index, cluster["representative_index"]])
            if current > best_iou:
                best_cluster, best_iou = cluster, current
        if best_cluster is not None and best_iou >= cluster_iou:
            best_cluster["members"].append(index)
            best_cluster["folds"].add(candidate["fold"])
        else:
            clusters.append({"representative_index": index, "members": [index], "folds": {candidate["fold"]}})
    output = []
    for cluster in clusters:
        if len(cluster["folds"]) < min_fold_support:
            continue
        representative = dict(candidates[cluster["representative_index"]])
        representative["fold_support"] = len(cluster["folds"])
        representative["consensus_score"] = float(np.mean([candidates[i]["quality_score"] for i in cluster["members"]]))
        output.append(representative)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge five M2/M3 test folds and create a calibrated Kaggle submission")
    add_path_args(parser)
    parser.add_argument("--method", choices=("m2", "m3"), required=True)
    parser.add_argument("--quality-model", type=Path, default=None)
    parser.add_argument("--quality-config", type=Path, default=None)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--cluster-iou", type=float, default=0.45)
    parser.add_argument("--min-fold-support", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--duplicate-iou", type=float, default=0.65)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    paths = paths_from_args(args)
    _, _, test_image_dir = discover_data(paths)
    quality_dir = paths.results_root / f"m4_quality_{args.method}"
    quality_model_path = (args.quality_model or quality_dir / "quality_model.joblib").resolve()
    quality_config_path = (args.quality_config or quality_dir / "best_quality.json").resolve()
    if not quality_model_path.is_file():
        raise FileNotFoundError(quality_model_path)
    if not quality_config_path.is_file():
        raise FileNotFoundError(quality_config_path)
    bundle = joblib.load(quality_model_path)
    model, feature_names = bundle["model"], bundle["features"]
    if list(feature_names) != FEATURES:
        raise RuntimeError(f"Quality feature mismatch: {feature_names} != {FEATURES}")
    quality_config = json.loads(quality_config_path.read_text())
    quality_threshold = float(args.quality_threshold if args.quality_threshold is not None else quality_config["quality_threshold"])
    min_area = int(args.min_area if args.min_area is not None else quality_config["min_area"])
    banks = [
        OOFProposalBank(paths.artifact_root / "predictions" / "sira_test" / args.method / f"fold{fold}")
        for fold in range(5)
    ]
    images = _test_images(test_image_dir)
    run_name = args.run_name or f"m4_{args.method}_ensemble_s{args.min_fold_support}"
    output_dir = paths.experiment_results(run_name)
    submission_rows, count_rows = [], []

    for image_path in tqdm(images, desc=f"Merge {args.method.upper()} test folds", unit="image"):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        candidates = []
        for fold, bank in enumerate(banks):
            bank_path = bank.path_for(image_path.name)
            if not bank_path.is_file():
                raise FileNotFoundError(f"Missing fold {fold} test prediction: {bank_path}")
            payload = bank.load(image_path.name, decode_masks=False)
            instances = payload["instances"]
            if not instances:
                continue
            frame = pd.DataFrame([{name: float(item.get("features", {}).get(name, 0.0)) for name in feature_names} for item in instances])
            scores = model.predict_proba(frame[feature_names])[:, 1]
            for item, score in zip(instances, scores):
                area = float(item.get("features", {}).get("area", 0.0))
                if score < quality_threshold or area < min_area:
                    continue
                candidates.append({
                    "fold": fold,
                    "quality_score": float(score),
                    "rle": rle_from_json(item["rle"]),
                })

        clustered = _cluster_candidates(candidates, args.cluster_iou, args.min_fold_support)
        if clustered:
            masks = np.stack([mask_utils.decode(item["rle"]).astype(bool) for item in clustered])
            scores = np.asarray([item["consensus_score"] for item in clustered], dtype=np.float32)
            selected = resolve_instances_scored(masks, scores, 0.0, min_area, args.duplicate_iou)
        else:
            selected = []
        for index, (mask, _) in enumerate(selected, start=1):
            submission_rows.append({
                "filament_id": f"{image_path.stem}_{index}",
                "segmentation_rle": mask_to_rle(mask),
            })
        count_rows.append({
            "file_name": image_path.name,
            "quality_candidates": len(candidates),
            "consensus_clusters": len(clustered),
            "instances": len(selected),
        })

    submission_path = output_dir / f"submission_{run_name}.csv"
    pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"]).to_csv(
        submission_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    pd.DataFrame(count_rows).to_csv(output_dir / "test_instance_counts.csv", index=False)
    write_json(submission_path.with_suffix(".metadata.json"), {
        "method": args.method,
        "quality_model": str(quality_model_path),
        "quality_model_sha256": sha256_file(quality_model_path),
        "quality_config": str(quality_config_path),
        "quality_threshold": quality_threshold,
        "min_area": min_area,
        "cluster_iou": args.cluster_iou,
        "min_fold_support": args.min_fold_support,
        "duplicate_iou": args.duplicate_iou,
        "images": len(images),
        "rows": len(submission_rows),
        "submission_sha256": sha256_file(submission_path),
    })
    print(f"Submission: {submission_path}")
    print(f"Rows: {len(submission_rows)}; images: {len(images)}")


if __name__ == "__main__":
    main()
