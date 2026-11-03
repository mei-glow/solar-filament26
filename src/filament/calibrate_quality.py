from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import add_path_args, discover_data, paths_from_args
from .oof_bank import OOFProposalBank
from .postprocess import mask_to_rle_object, polygon_to_mask, pq_stats_rle, resolve_instances_scored, rle_from_json


FEATURES = ["detector_score", "mean_probability", "q90_probability", "seed_agreement", "semantic_support",
            "boundary_uncertainty", "elongation", "radial_position", "area", "components"]


def _model(kind: str):
    if kind == "logistic": return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=.04, max_leaf_nodes=15, l2_regularization=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="M4 group-OOF mask quality calibration and pooled-PQ threshold tuning")
    add_path_args(parser); parser.add_argument("--run-name", required=True); parser.add_argument("--sira-run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", choices=("logistic", "histgb"), default="logistic")
    parser.add_argument("--threshold-values", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--min-area-values", default="100,200,300")
    args = parser.parse_args(); paths = paths_from_args(args); annotation_path, _, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve())
    rows, payload_path_by_file = [], {}
    root = paths.artifact_root / "predictions" / "sira_oof" / args.sira_run_name
    for fold in range(args.folds):
        bank = OOFProposalBank(root / f"fold{fold}")
        for path in bank.files():
            payload = bank.load(path, decode_masks=False); payload_path_by_file[payload["file_name"]] = path
            for index, instance in enumerate(payload["instances"]):
                values = instance.get("features", {})
                rows.append({"file_name": payload["file_name"], "fold": fold, "candidate_index": index,
                             **{key: float(values.get(key, 0.0)) for key in FEATURES}, "target_iou": float(values.get("target_iou", 0.0))})
    frame = pd.DataFrame(rows)
    if frame.empty: raise RuntimeError(f"No refined OOF candidates under {root}")
    labels = (frame["target_iou"].to_numpy() > .5).astype(int)
    if len(np.unique(labels)) < 2: raise RuntimeError("Quality labels contain only one class")
    groups = frame["file_name"].to_numpy(); splits = min(5, len(np.unique(groups)))
    estimator = _model(args.model)
    oof_probability = cross_val_predict(estimator, frame[FEATURES], labels, groups=groups,
                                        cv=GroupKFold(splits), method="predict_proba")[:, 1]
    frame["quality_probability"] = oof_probability
    estimator.fit(frame[FEATURES], labels)
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(x["id"]): x for x in coco["images"]}; anns = defaultdict(list)
    for item in coco["annotations"]: anns[str(item["image_id"])].append(item)
    thresholds = [float(x) for x in args.threshold_values.split(",")]; areas = [int(x) for x in args.min_area_values.split(",")]
    stats = {(t, a): dict(iou_sum=0., pq_sum=0., count=0, tp=0, fp=0, fn=0) for t in thresholds for a in areas}
    quality_lookup = {(row.file_name, int(row.candidate_index)): float(row.quality_probability) for row in frame.itertuples(index=False)}
    for file_name, records in manifest.groupby("file_name", sort=True):
        payload_path = payload_path_by_file.get(file_name)
        if payload_path is None: raise FileNotFoundError(f"Refined OOF bank missing {file_name}")
        payload = OOFProposalBank(payload_path.parent).load(payload_path, decode_masks=False)
        if payload["instances"]:
            decoded = mask_utils.decode([rle_from_json(item["rle"]) for item in payload["instances"]])
            masks = np.moveaxis(decoded, -1, 0).astype(bool, copy=False)
        else:
            masks = np.empty((0, 1, 1), bool)
        scores = np.asarray([quality_lookup[(file_name, i)] for i in range(len(payload["instances"]))], np.float32)
        for threshold in thresholds:
            for area in areas:
                selected = resolve_instances_scored(masks, scores, threshold, area)
                pred = [mask_to_rle_object(mask) for mask, _ in selected]; current = stats[(threshold, area)]
                for record in records.itertuples(index=False):
                    info = images[str(record.image_id)]
                    gt = [mask_to_rle_object(polygon_to_mask(x["segmentation"], info["height"], info["width"])) for x in anns[str(record.image_id)]]
                    values = pq_stats_rle(gt, pred)
                    for key in ("iou_sum", "tp", "fp", "fn"): current[key] += values[key]
                    current["pq_sum"] += values["pq"]; current["count"] += 1
    ranking = []
    for (threshold, area), values in stats.items():
        denominator = values["tp"] + .5 * values["fp"] + .5 * values["fn"]
        ranking.append({"quality_threshold": threshold, "min_area": area,
                        "global_pq": values["iou_sum"] / denominator if denominator else 0., "mean_pq": values["pq_sum"] / max(values["count"], 1),
                        "tp": values["tp"], "fp": values["fp"], "fn": values["fn"]})
    ranking = pd.DataFrame(ranking).sort_values("global_pq", ascending=False)
    output = paths.experiment_results(args.run_name); frame.to_csv(output / "oof_quality_predictions.csv", index=False)
    ranking.to_csv(output / "quality_tuning.csv", index=False); joblib.dump({"model": estimator, "features": FEATURES}, output / "quality_model.joblib")
    (output / "best_quality.json").write_text(json.dumps(ranking.iloc[0].to_dict(), indent=2) + "\n")
    print(ranking.head(12).to_string(index=False)); print(f"Saved calibrated model: {output / 'quality_model.joblib'}")


if __name__ == "__main__": main()
