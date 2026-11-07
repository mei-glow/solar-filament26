from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import write_json
from .gated_selection import FEATURES, build_candidate_features, choose_candidate_rles, select_set_aware_instances
from .oof_bank import OOFProposalBank
from .postprocess import polygon_to_rle, pq_stats_rle, rle_from_json


def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=220, learning_rate=0.04, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=1.0, random_state=3020,
    )


def _assigned_iou(pred_rles: list[dict], gt_rles: list[dict]) -> np.ndarray:
    targets = np.zeros(len(pred_rles), np.float32)
    if not pred_rles or not gt_rles:
        return targets
    ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    pred_indices, gt_indices = linear_sum_assignment(-ious)
    targets[pred_indices] = ious[pred_indices, gt_indices]
    return targets


def _parse_values(text: str, cast):
    return [cast(value) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="P0/P1/P2 gated raw-refined selection with unique Hungarian OOF targets")
    add_path_args(parser)
    parser.add_argument("--run-name", default="m5_p0p1p2_m2")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw-bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--sira-run-name", default="m2_sira_oof")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--action-modes", default="raw,refined,gated")
    parser.add_argument("--refine-margins", default="0.00,0.05")
    parser.add_argument("--score-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--cluster-ious", default="0.30,0.45")
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--duplicate-iou", type=float, default=0.65)
    args = parser.parse_args()

    paths = paths_from_args(args); annotation_path, _, _ = discover_data(paths)
    manifest_path = (args.manifest or paths.results_root / "fold_manifest_s3020.csv").resolve()
    manifest = pd.read_csv(manifest_path)
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]: annotations[str(item["image_id"])].append(item)

    rows, payloads = [], {}
    grouped = list(manifest.groupby("file_name", sort=True))
    for file_name, records in tqdm(grouped, desc="Build P0/P1 candidate targets", unit="image"):
        fold = int(records["fold"].iloc[0])
        raw = OOFProposalBank(paths.artifact_root / "predictions" / "oof" / args.raw_bank_name / f"fold{fold}").load(file_name, decode_masks=False)
        refined = OOFProposalBank(paths.artifact_root / "predictions" / "sira_oof" / args.sira_run_name / f"fold{fold}").load(file_name, decode_masks=False)
        raw_instances, refined_instances = raw["instances"], refined["instances"]
        features = build_candidate_features(raw_instances, refined_instances)
        raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
        refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
        raw_target = np.zeros(len(raw_rles), np.float32); refined_target = np.zeros(len(raw_rles), np.float32)
        for record in records.itertuples(index=False):
            image_id = str(record.image_id); info = images[image_id]
            gt = [polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]]
            raw_target += _assigned_iou(raw_rles, gt); refined_target += _assigned_iou(refined_rles, gt)
        raw_target /= len(records); refined_target /= len(records)
        for index, feature_row in features.iterrows():
            rows.append({
                "file_name": file_name, "fold": fold, "candidate_index": int(index),
                **{name: float(feature_row[name]) for name in FEATURES},
                "raw_unique_iou_target": float(raw_target[index]),
                "refined_unique_iou_target": float(refined_target[index]),
            })
        payloads[file_name] = (records, raw_instances, refined_instances)

    frame = pd.DataFrame(rows)
    groups = frame["file_name"].to_numpy(); splits = min(args.folds, frame["file_name"].nunique())
    raw_oof = np.zeros(len(frame), np.float32); refined_oof = np.zeros(len(frame), np.float32)
    for split_index, (train_index, validation_index) in enumerate(GroupKFold(splits).split(frame, groups=groups)):
        raw_model, refined_model = _regressor(), _regressor()
        raw_model.fit(frame.iloc[train_index][FEATURES], frame.iloc[train_index]["raw_unique_iou_target"])
        refined_model.fit(frame.iloc[train_index][FEATURES], frame.iloc[train_index]["refined_unique_iou_target"])
        raw_oof[validation_index] = raw_model.predict(frame.iloc[validation_index][FEATURES])
        refined_oof[validation_index] = refined_model.predict(frame.iloc[validation_index][FEATURES])
        print(f"Cross-fit split {split_index + 1}/{splits}: validation candidates={len(validation_index)}", flush=True)
    frame["predicted_raw_unique_iou"] = np.clip(raw_oof, 0, 1)
    frame["predicted_refined_unique_iou"] = np.clip(refined_oof, 0, 1)

    raw_model, refined_model = _regressor(), _regressor()
    raw_model.fit(frame[FEATURES], frame["raw_unique_iou_target"])
    refined_model.fit(frame[FEATURES], frame["refined_unique_iou_target"])

    action_modes = _parse_values(args.action_modes, str)
    margins = _parse_values(args.refine_margins, float)
    thresholds = _parse_values(args.score_thresholds, float)
    cluster_ious = _parse_values(args.cluster_ious, float)
    configurations = []
    for mode in action_modes:
        mode_margins = margins if mode == "gated" else [0.0]
        for margin in mode_margins:
            for cluster_iou in cluster_ious:
                configurations.append((mode, margin, cluster_iou))
    stats = {
        (mode, margin, cluster_iou, threshold): dict(iou_sum=0.0, pq_sum=0.0, count=0, tp=0, fp=0, fn=0, refined=0, candidates=0)
        for mode, margin, cluster_iou in configurations for threshold in thresholds
    }

    by_file = {name: values.sort_values("candidate_index") for name, values in frame.groupby("file_name", sort=False)}
    minimum_threshold = min(thresholds)
    for file_name, (records, raw_instances, refined_instances) in tqdm(payloads.items(), desc="P0/P1/P2 official OOF sweep", unit="image"):
        candidate_frame = by_file.get(file_name)
        if candidate_frame is None:
            predicted_raw = np.empty(0, np.float32); predicted_refined = np.empty(0, np.float32)
        else:
            predicted_raw = candidate_frame["predicted_raw_unique_iou"].to_numpy(np.float32)
            predicted_refined = candidate_frame["predicted_refined_unique_iou"].to_numpy(np.float32)
        gt_by_record = []
        for record in records.itertuples(index=False):
            image_id = str(record.image_id); info = images[image_id]
            gt_by_record.append([polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]])
        for mode, margin, cluster_iou in configurations:
            chosen_rles, scores, use_refined = choose_candidate_rles(
                raw_instances, refined_instances, predicted_raw, predicted_refined, mode, margin,
            )
            resolved = select_set_aware_instances(
                chosen_rles, scores, minimum_threshold, args.min_area, cluster_iou, args.duplicate_iou,
            )
            for threshold in thresholds:
                predictions = [rle for rle, score in resolved if score >= threshold]
                current = stats[(mode, margin, cluster_iou, threshold)]
                current["refined"] += int(use_refined[scores >= threshold].sum())
                current["candidates"] += int((scores >= threshold).sum())
                for gt in gt_by_record:
                    values = pq_stats_rle(gt, predictions)
                    for key in ("iou_sum", "tp", "fp", "fn"): current[key] += values[key]
                    current["pq_sum"] += values["pq"]; current["count"] += 1

    ranking_rows = []
    for (mode, margin, cluster_iou, threshold), values in stats.items():
        denominator = values["tp"] + 0.5 * values["fp"] + 0.5 * values["fn"]
        ranking_rows.append({
            "action_mode": mode, "refine_margin": margin, "score_threshold": threshold,
            "cluster_iou": cluster_iou, "min_area": args.min_area, "duplicate_iou": args.duplicate_iou,
            "global_pq": values["iou_sum"] / denominator if denominator else 0.0,
            "mean_pq": values["pq_sum"] / max(values["count"], 1),
            "sq": values["iou_sum"] / max(values["tp"], 1),
            "rq": values["tp"] / denominator if denominator else 0.0,
            "tp": values["tp"], "fp": values["fp"], "fn": values["fn"],
            "refined_fraction": values["refined"] / max(values["candidates"], 1),
        })
    ranking = pd.DataFrame(ranking_rows).sort_values("global_pq", ascending=False)
    output = paths.experiment_results(args.run_name)
    frame.to_csv(output / "oof_gate_candidates.csv", index=False)
    ranking.to_csv(output / "gate_setaware_tuning.csv", index=False)
    best = ranking.iloc[0].to_dict(); write_json(output / "best_gate_setaware.json", best)
    joblib.dump({"raw_model": raw_model, "refined_model": refined_model, "features": FEATURES}, output / "gate_ranker.joblib")
    write_json(output / "metadata.json", {
        "manifest": str(manifest_path), "raw_bank_name": args.raw_bank_name,
        "sira_run_name": args.sira_run_name, "candidate_rows": len(frame),
        "target": "mean one-to-one Hungarian assigned IoU across annotation records",
    })
    print(ranking.head(20).to_string(index=False), flush=True)
    print(f"Best P0/P1/P2: {output / 'best_gate_setaware.json'}", flush=True)
    print(f"Final gate/ranker: {output / 'gate_ranker.joblib'}", flush=True)


if __name__ == "__main__":
    main()
