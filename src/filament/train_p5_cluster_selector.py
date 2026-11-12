from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import write_json
from .oof_bank import OOFProposalBank
from .p5_selector import decode_cluster_first
from .postprocess import polygon_to_rle, rle_from_json
from .train_p5_selector import _empty_stats, _metrics, _parse_values, _regressor, _update_stats


def _fit_variant_models(frame: pd.DataFrame, features: list[str], seed: int):
    quality = frame[["raw_anchor_iou_mean", "refined_same_anchor_iou_mean"]].max(axis=1).to_numpy(np.float64)
    weights = 0.5 + 1.5 * quality
    raw_model = _regressor(seed)
    refined_model = _regressor(seed + 1)
    raw_model.fit(frame[features], frame["raw_anchor_iou_mean"], sample_weight=weights)
    refined_model.fit(frame[features], frame["refined_same_anchor_iou_mean"], sample_weight=weights)
    return raw_model, refined_model


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.3 cluster-first representative and paired-variant selector")
    add_path_args(parser)
    parser.add_argument("--run-name", default="p5_3_cluster_first_m2")
    parser.add_argument("--p5-run", default="p5_2_graph_selector_m2")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw-bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--sira-run-name", default="m2_sira_oof")
    parser.add_argument("--quality-thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--duplicate-thresholds", default="0.30,0.50,0.70,0.85,1.01")
    parser.add_argument("--min-area-values", default="25,100,200,300")
    args = parser.parse_args()

    paths = paths_from_args(args)
    annotation_path, _, _ = discover_data(paths)
    p5_root = paths.results_root / args.p5_run
    candidate_path = p5_root / "oof_candidate_predictions.csv"
    pair_path = p5_root / "oof_pair_predictions.csv.gz"
    p5_model_path = p5_root / "p5_selector.joblib"
    for path in (candidate_path, pair_path, p5_model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = pd.read_csv(candidate_path)
    pairs = pd.read_csv(pair_path)
    p5_bundle = joblib.load(p5_model_path)
    feature_names = list(p5_bundle["candidate_features"])
    forbidden = ("oracle_", "target_", "raw_anchor", "refined_same_anchor", "same_gt")
    if any(name.startswith(forbidden) for name in feature_names):
        raise RuntimeError("Target leakage in P5.3 features")

    raw_oof = np.zeros(len(candidates), np.float64)
    refined_oof = np.zeros(len(candidates), np.float64)
    folds = sorted(int(value) for value in candidates["fold"].unique())
    for fold in folds:
        train = candidates["fold"].to_numpy() != fold
        valid = ~train
        raw_model, refined_model = _fit_variant_models(candidates.loc[train], feature_names, 7020 + fold * 10)
        raw_oof[valid] = np.clip(raw_model.predict(candidates.loc[valid, feature_names]), 0.0, 1.0)
        refined_oof[valid] = np.clip(refined_model.predict(candidates.loc[valid, feature_names]), 0.0, 1.0)
        print(f"P5.3 cross-fit fold={fold}: candidates={int(valid.sum())}", flush=True)
    candidates["predicted_raw_paired_iou"] = raw_oof
    candidates["predicted_refined_paired_iou"] = refined_oof
    full_raw_model, full_refined_model = _fit_variant_models(candidates, feature_names, 8020)

    manifest_path = (args.manifest or paths.results_root / "fold_manifest_s3020.csv").resolve()
    manifest = pd.read_csv(manifest_path)
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]:
        annotations[str(item["image_id"])].append(item)
    candidate_by_file = {
        name: group.sort_values("candidate_index") for name, group in candidates.groupby("file_name", sort=False)
    }
    pair_by_file = {name: group for name, group in pairs.groupby("file_name", sort=False)}

    image_cache = []
    for file_name, records in tqdm(list(manifest.groupby("file_name", sort=True)), desc="Cache P5.3 OOF IoUs", unit="image"):
        fold = int(records["fold"].iloc[0])
        raw_payload = OOFProposalBank(
            paths.artifact_root / "predictions" / "oof" / args.raw_bank_name / f"fold{fold}"
        ).load(file_name, decode_masks=False)
        refined_payload = OOFProposalBank(
            paths.artifact_root / "predictions" / "sira_oof" / args.sira_run_name / f"fold{fold}"
        ).load(file_name, decode_masks=False)
        raw_rles = [rle_from_json(item["rle"]) for item in raw_payload["instances"]]
        refined_rles = [rle_from_json(item["rle"]) for item in refined_payload["instances"]]
        current_candidates = candidate_by_file.get(file_name, candidates.iloc[0:0].copy())
        if len(raw_rles) != len(refined_rles) or len(raw_rles) != len(current_candidates):
            raise RuntimeError(f"Candidate alignment mismatch for {file_name}")
        raw_matrices, refined_matrices = [], []
        for record in records.itertuples(index=False):
            image_id = str(record.image_id)
            info = images[image_id]
            gt = [polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]]
            raw_matrices.append(
                mask_utils.iou(raw_rles, gt, [0] * len(gt)) if raw_rles and gt else np.zeros((len(raw_rles), len(gt)), np.float32)
            )
            refined_matrices.append(
                mask_utils.iou(refined_rles, gt, [0] * len(gt)) if refined_rles and gt else np.zeros((len(raw_rles), len(gt)), np.float32)
            )
        image_cache.append((
            file_name,
            fold,
            current_candidates,
            pair_by_file.get(file_name, pairs.iloc[0:0]),
            raw_matrices,
            refined_matrices,
        ))

    quality_thresholds = _parse_values(args.quality_thresholds, float)
    duplicate_thresholds = _parse_values(args.duplicate_thresholds, float)
    min_areas = _parse_values(args.min_area_values, int)
    configurations = [
        (quality, duplicate, area)
        for quality in quality_thresholds
        for duplicate in duplicate_thresholds
        for area in min_areas
    ]
    stats = {configuration: _empty_stats() for configuration in configurations}
    for _, _, current_candidates, current_pairs, raw_matrices, refined_matrices in tqdm(
        image_cache, desc="P5.3 cluster-first sweep", unit="image"
    ):
        for quality_threshold, duplicate_threshold, min_area in configurations:
            selected = decode_cluster_first(
                current_candidates,
                current_pairs,
                quality_threshold,
                duplicate_threshold,
                min_area,
            )
            current = stats[(quality_threshold, duplicate_threshold, min_area)]
            current["selected"] += len(selected)
            current["raw"] += sum(item["action"] == "raw" for item in selected)
            current["refined"] += sum(item["action"] == "refined" for item in selected)
            for raw_iou, refined_iou in zip(raw_matrices, refined_matrices):
                _update_stats(current, selected, raw_iou, refined_iou)

    ranking_rows = []
    for (quality_threshold, duplicate_threshold, min_area), values in stats.items():
        ranking_rows.append({
            "quality_threshold": quality_threshold,
            "duplicate_threshold": duplicate_threshold,
            "min_area": min_area,
            **_metrics(values),
            "selected_candidates": int(values["selected"]),
            "raw_fraction": values["raw"] / max(values["selected"], 1),
            "refined_fraction": values["refined"] / max(values["selected"], 1),
        })
    ranking = pd.DataFrame(ranking_rows).sort_values("global_pq", ascending=False).reset_index(drop=True)
    best = ranking.iloc[0].to_dict()

    fold_values = {fold: _empty_stats() for fold in folds}
    for _, fold, current_candidates, current_pairs, raw_matrices, refined_matrices in image_cache:
        selected = decode_cluster_first(
            current_candidates,
            current_pairs,
            float(best["quality_threshold"]),
            float(best["duplicate_threshold"]),
            int(best["min_area"]),
        )
        values = fold_values[fold]
        values["selected"] += len(selected)
        values["raw"] += sum(item["action"] == "raw" for item in selected)
        values["refined"] += sum(item["action"] == "refined" for item in selected)
        for raw_iou, refined_iou in zip(raw_matrices, refined_matrices):
            _update_stats(values, selected, raw_iou, refined_iou)
    fold_rows = [{"fold": fold, **_metrics(values), "selected_candidates": values["selected"]} for fold, values in fold_values.items()]

    true_raw = candidates["raw_anchor_iou_mean"].to_numpy(np.float64)
    true_refined = candidates["refined_same_anchor_iou_mean"].to_numpy(np.float64)
    predicted_quality = np.maximum(raw_oof, refined_oof)
    true_quality = np.maximum(true_raw, true_refined)
    matched = true_quality > 0.5
    stable_variant = np.abs(true_raw - true_refined) >= 0.02
    oracle_selected = candidates["oracle_selected"].astype(bool).to_numpy()
    oracle_variant = candidates["oracle_action"].to_numpy()
    predicted_variant = np.where(raw_oof >= refined_oof, "raw", "refined")
    diagnostics = {
        "raw_iou_mae": float(mean_absolute_error(true_raw, raw_oof)),
        "refined_iou_mae": float(mean_absolute_error(true_refined, refined_oof)),
        "match_roc_auc": float(roc_auc_score(matched, predicted_quality)),
        "match_pr_auc": float(average_precision_score(matched, predicted_quality)),
        "match_base_rate": float(matched.mean()),
        "variant_sign_accuracy_stable": float((predicted_variant[stable_variant] == np.where(true_raw[stable_variant] >= true_refined[stable_variant], "raw", "refined")).mean()),
        "variant_action_accuracy_oracle_selected": float((predicted_variant[oracle_selected] == oracle_variant[oracle_selected]).mean()),
    }

    output = paths.experiment_results(args.run_name)
    candidates.to_csv(output / "oof_cluster_candidates.csv", index=False)
    ranking.to_csv(output / "p5_3_tuning.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "best_by_fold.csv", index=False)
    write_json(output / "best_p5_cluster_selector.json", best)
    write_json(output / "diagnostics.json", diagnostics)
    joblib.dump({
        "raw_iou_model": full_raw_model,
        "refined_iou_model": full_refined_model,
        "pair_model": p5_bundle["pair_model"],
        "candidate_features": feature_names,
        "pair_features": p5_bundle["pair_features"],
        "pair_prefilter": p5_bundle["pair_prefilter"],
    }, output / "p5_cluster_selector.joblib")
    write_json(output / "metadata.json", {
        "source_p5_run": args.p5_run,
        "manifest": str(manifest_path),
        "cross_fit_folds": folds,
        "candidate_features": feature_names,
        "targets": ["raw_anchor_iou_mean", "refined_same_anchor_iou_mean"],
        "decoder": "duplicate components first, best predicted paired-IoU representative second, cluster threshold last",
    })
    print("\nP5.3 diagnostics:", json.dumps(diagnostics, indent=2), flush=True)
    print("\nP5.3 top configurations:", flush=True)
    print(ranking.head(20).to_string(index=False), flush=True)
    print("\nBest configuration by fold:", flush=True)
    print(pd.DataFrame(fold_rows).to_string(index=False), flush=True)
    print(f"Best P5.3 configuration: {output / 'best_p5_cluster_selector.json'}", flush=True)


if __name__ == "__main__":
    main()
