from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, roc_auc_score
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import write_json
from .gated_selection import FEATURES
from .oof_bank import OOFProposalBank
from .oracle_decomposition import _one_to_one_stats
from .p5_selector import (
    ACTIONS,
    CONTEXT_FEATURES,
    PAIR_FEATURES,
    add_candidate_context,
    decode_candidate_set,
    normalize_action_predictions,
    pair_prefilter,
)
from .postprocess import polygon_to_rle, rle_from_json


TARGET_COLUMNS = ["target_p_reject", "target_p_raw", "target_p_refined"]


def _parse_values(text: str, cast) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def _regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=240,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    )


def _action_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["oracle_action"].value_counts()
    class_weight = {name: len(frame) / (len(ACTIONS) * int(counts[name])) for name in ACTIONS}
    balance = frame["oracle_action"].map(class_weight).to_numpy(np.float64)
    confidence = 0.5 + 0.5 * np.clip(frame["oracle_action_margin"].to_numpy(np.float64) / 0.05, 0.0, 1.0)
    return balance * confidence


def _fit_action_models(
    frame: pd.DataFrame,
    feature_names: list[str],
    seed: int,
) -> list[HistGradientBoostingRegressor]:
    weights = _action_weights(frame)
    models = []
    for index, target in enumerate(TARGET_COLUMNS):
        model = _regressor(seed + index)
        model.fit(frame[feature_names], frame[target], sample_weight=weights)
        models.append(model)
    return models


def _predict_actions(models: list[HistGradientBoostingRegressor], frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = np.column_stack([model.predict(frame[features]) for model in models])
    return normalize_action_predictions(values)


def _update_stats(current: dict, selected: list[dict], raw_iou: np.ndarray, refined_iou: np.ndarray) -> None:
    if selected:
        rows = [
            (raw_iou if item["action"] == "raw" else refined_iou)[int(item["candidate_index"])]
            for item in selected
        ]
        matrix = np.asarray(rows, np.float32)
    else:
        matrix = np.zeros((0, raw_iou.shape[1]), np.float32)
    values = _one_to_one_stats(matrix, len(selected), raw_iou.shape[1])
    denominator = values["tp"] + 0.5 * values["fp"] + 0.5 * values["fn"]
    for key in ("iou_sum", "tp", "fp", "fn"):
        current[key] += values[key]
    current["pq_sum"] += values["iou_sum"] / denominator if denominator else 0.0
    current["records"] += 1


def _metrics(values: dict) -> dict:
    denominator = values["tp"] + 0.5 * values["fp"] + 0.5 * values["fn"]
    return {
        "global_pq": values["iou_sum"] / denominator if denominator else 0.0,
        "mean_pq": values["pq_sum"] / max(values["records"], 1),
        "sq": values["iou_sum"] / max(values["tp"], 1),
        "rq": values["tp"] / denominator if denominator else 0.0,
        "tp": int(values["tp"]),
        "fp": int(values["fp"]),
        "fn": int(values["fn"]),
    }


def _empty_stats() -> dict:
    return dict(iou_sum=0.0, pq_sum=0.0, records=0, tp=0, fp=0, fn=0, selected=0, raw=0, refined=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.2 cross-fitted action, duplicate, and graph-set selector")
    add_path_args(parser)
    parser.add_argument("--run-name", default="p5_2_graph_selector_m2")
    parser.add_argument("--supervision-run", default="p5_1_oracle_supervision_m2")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw-bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--sira-run-name", default="m2_sira_oof")
    parser.add_argument("--pair-prefilter", type=float, default=0.05)
    parser.add_argument("--selection-thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90")
    parser.add_argument("--duplicate-thresholds", default="0.20,0.30,0.50,0.70,0.85,1.01")
    parser.add_argument("--min-area-values", default="25,100,200,300")
    parser.add_argument("--decoder-modes", default="edge,component")
    args = parser.parse_args()

    paths = paths_from_args(args)
    annotation_path, _, _ = discover_data(paths)
    supervision = paths.results_root / args.supervision_run
    candidate_path = supervision / "candidate_supervision.csv"
    pair_path = supervision / "pair_supervision.csv.gz"
    if not candidate_path.is_file() or not pair_path.is_file():
        raise FileNotFoundError(f"Missing P5.1 supervision under {supervision}")
    candidates = pd.read_csv(candidate_path)
    all_pairs = pd.read_csv(pair_path)
    candidates = add_candidate_context(candidates, all_pairs)
    candidate_features = FEATURES + CONTEXT_FEATURES
    if any(name.startswith(("oracle_", "target_", "raw_anchor", "refined_same_anchor", "same_gt")) for name in candidate_features):
        raise RuntimeError("Target leakage in P5.2 candidate features")
    missing = [name for name in candidate_features + TARGET_COLUMNS if name not in candidates]
    if missing:
        raise RuntimeError(f"Missing candidate columns: {missing}")

    eligible_pairs = all_pairs.loc[pair_prefilter(all_pairs, args.pair_prefilter)].copy()
    missing_pair = [name for name in PAIR_FEATURES + ["same_gt_frequency_03"] if name not in eligible_pairs]
    if missing_pair:
        raise RuntimeError(f"Missing pair columns: {missing_pair}")
    folds = sorted(int(value) for value in candidates["fold"].unique())
    if folds != sorted(int(value) for value in eligible_pairs["fold"].unique()):
        raise RuntimeError("Candidate and pair folds differ")

    action_oof = np.zeros((len(candidates), len(ACTIONS)), np.float64)
    pair_oof = np.zeros(len(eligible_pairs), np.float64)
    for fold in folds:
        train_candidates = candidates["fold"].to_numpy() != fold
        valid_candidates = ~train_candidates
        models = _fit_action_models(candidates.loc[train_candidates], candidate_features, 3020 + fold * 10)
        action_oof[valid_candidates] = _predict_actions(models, candidates.loc[valid_candidates], candidate_features)

        train_pairs = eligible_pairs["fold"].to_numpy() != fold
        valid_pairs = ~train_pairs
        pair_model = _regressor(4020 + fold)
        pair_target = eligible_pairs.loc[train_pairs, "same_gt_frequency_03"].to_numpy(np.float64)
        pair_weight = 0.5 + np.abs(pair_target - 0.5)
        pair_model.fit(eligible_pairs.loc[train_pairs, PAIR_FEATURES], pair_target, sample_weight=pair_weight)
        pair_oof[valid_pairs] = np.clip(pair_model.predict(eligible_pairs.loc[valid_pairs, PAIR_FEATURES]), 0.0, 1.0)
        print(
            f"P5.2 cross-fit fold={fold}: candidates={int(valid_candidates.sum())}, pairs={int(valid_pairs.sum())}",
            flush=True,
        )

    for index, action in enumerate(ACTIONS):
        candidates[f"predicted_p_{action}"] = action_oof[:, index]
    eligible_pairs["predicted_duplicate_probability"] = pair_oof

    full_action_models = _fit_action_models(candidates, candidate_features, 5020)
    full_pair_model = _regressor(6020)
    full_pair_target = eligible_pairs["same_gt_frequency_03"].to_numpy(np.float64)
    full_pair_model.fit(
        eligible_pairs[PAIR_FEATURES],
        full_pair_target,
        sample_weight=0.5 + np.abs(full_pair_target - 0.5),
    )

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
    pair_by_file = {name: group for name, group in eligible_pairs.groupby("file_name", sort=False)}
    image_cache = []
    for file_name, records in tqdm(list(manifest.groupby("file_name", sort=True)), desc="Cache P5.2 OOF IoUs", unit="image"):
        fold = int(records["fold"].iloc[0])
        raw_payload = OOFProposalBank(
            paths.artifact_root / "predictions" / "oof" / args.raw_bank_name / f"fold{fold}"
        ).load(file_name, decode_masks=False)
        refined_payload = OOFProposalBank(
            paths.artifact_root / "predictions" / "sira_oof" / args.sira_run_name / f"fold{fold}"
        ).load(file_name, decode_masks=False)
        raw_rles = [rle_from_json(item["rle"]) for item in raw_payload["instances"]]
        refined_rles = [rle_from_json(item["rle"]) for item in refined_payload["instances"]]
        current_candidates = candidate_by_file[file_name]
        if len(raw_rles) != len(refined_rles) or len(raw_rles) != len(current_candidates):
            raise RuntimeError(f"Candidate alignment mismatch for {file_name}")
        gt_sets, raw_matrices, refined_matrices = [], [], []
        for record in records.itertuples(index=False):
            image_id = str(record.image_id)
            info = images[image_id]
            gt = [polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]]
            gt_sets.append(gt)
            raw_matrices.append(
                mask_utils.iou(raw_rles, gt, [0] * len(gt)) if raw_rles and gt else np.zeros((len(raw_rles), len(gt)), np.float32)
            )
            refined_matrices.append(
                mask_utils.iou(refined_rles, gt, [0] * len(gt)) if refined_rles and gt else np.zeros((len(raw_rles), len(gt)), np.float32)
            )
        current_pairs = pair_by_file.get(file_name, eligible_pairs.iloc[0:0])
        image_cache.append((file_name, fold, current_candidates, current_pairs, raw_matrices, refined_matrices))

    selection_thresholds = _parse_values(args.selection_thresholds, float)
    duplicate_thresholds = _parse_values(args.duplicate_thresholds, float)
    min_areas = _parse_values(args.min_area_values, int)
    decoder_modes = _parse_values(args.decoder_modes, str)
    configurations = [
        (mode, selection, duplicate, area)
        for mode in decoder_modes
        for selection in selection_thresholds
        for duplicate in duplicate_thresholds
        for area in min_areas
    ]
    stats = {configuration: _empty_stats() for configuration in configurations}
    for _, _, current_candidates, current_pairs, raw_matrices, refined_matrices in tqdm(
        image_cache, desc="P5.2 graph decoder sweep", unit="image"
    ):
        for configuration in configurations:
            mode, selection_threshold, duplicate_threshold, min_area = configuration
            selected = decode_candidate_set(
                current_candidates,
                current_pairs,
                selection_threshold,
                duplicate_threshold,
                min_area,
                mode,
            )
            current = stats[configuration]
            current["selected"] += len(selected)
            current["raw"] += sum(item["action"] == "raw" for item in selected)
            current["refined"] += sum(item["action"] == "refined" for item in selected)
            for raw_iou, refined_iou in zip(raw_matrices, refined_matrices):
                _update_stats(current, selected, raw_iou, refined_iou)

    ranking_rows = []
    for (mode, selection_threshold, duplicate_threshold, min_area), values in stats.items():
        row = {
            "decoder_mode": mode,
            "selection_threshold": selection_threshold,
            "duplicate_threshold": duplicate_threshold,
            "min_area": min_area,
            **_metrics(values),
            "selected_candidates": int(values["selected"]),
            "raw_fraction": values["raw"] / max(values["selected"], 1),
            "refined_fraction": values["refined"] / max(values["selected"], 1),
        }
        ranking_rows.append(row)
    ranking = pd.DataFrame(ranking_rows).sort_values("global_pq", ascending=False).reset_index(drop=True)
    best = ranking.iloc[0].to_dict()

    fold_values = {fold: _empty_stats() for fold in folds}
    for _, fold, current_candidates, current_pairs, raw_matrices, refined_matrices in image_cache:
        selected = decode_candidate_set(
            current_candidates,
            current_pairs,
            float(best["selection_threshold"]),
            float(best["duplicate_threshold"]),
            int(best["min_area"]),
            str(best["decoder_mode"]),
        )
        fold_values[fold]["selected"] += len(selected)
        fold_values[fold]["raw"] += sum(item["action"] == "raw" for item in selected)
        fold_values[fold]["refined"] += sum(item["action"] == "refined" for item in selected)
        for raw_iou, refined_iou in zip(raw_matrices, refined_matrices):
            _update_stats(fold_values[fold], selected, raw_iou, refined_iou)
    fold_rows = [{"fold": fold, **_metrics(values), "selected_candidates": values["selected"]} for fold, values in fold_values.items()]

    hard_target = candidates["oracle_action"].map({name: index for index, name in enumerate(ACTIONS)}).to_numpy()
    hard_prediction = action_oof.argmax(axis=1)
    selected_target = hard_target != 0
    selected_score = 1.0 - action_oof[:, 0]
    pair_binary = (eligible_pairs["same_gt_frequency_03"].to_numpy() > 0).astype(int)
    diagnostics = {
        "candidate_action_accuracy": float((hard_target == hard_prediction).mean()),
        "candidate_action_log_loss": float(log_loss(hard_target, action_oof, labels=list(range(len(ACTIONS))))),
        "candidate_selection_roc_auc": float(roc_auc_score(selected_target, selected_score)),
        "candidate_selection_pr_auc": float(average_precision_score(selected_target, selected_score)),
        "candidate_selection_base_rate": float(selected_target.mean()),
        "pair_mae": float(mean_absolute_error(eligible_pairs["same_gt_frequency_03"], pair_oof)),
        "pair_roc_auc": float(roc_auc_score(pair_binary, pair_oof)),
        "pair_pr_auc": float(average_precision_score(pair_binary, pair_oof)),
        "pair_positive_rate": float(pair_binary.mean()),
        "eligible_pairs": int(len(eligible_pairs)),
    }

    output = paths.experiment_results(args.run_name)
    candidates.to_csv(output / "oof_candidate_predictions.csv", index=False)
    eligible_pairs.to_csv(output / "oof_pair_predictions.csv.gz", index=False, compression="gzip")
    ranking.to_csv(output / "p5_2_tuning.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "best_by_fold.csv", index=False)
    write_json(output / "best_p5_selector.json", best)
    write_json(output / "diagnostics.json", diagnostics)
    joblib.dump(
        {
            "action_models": full_action_models,
            "pair_model": full_pair_model,
            "candidate_features": candidate_features,
            "pair_features": PAIR_FEATURES,
            "actions": ACTIONS,
            "pair_prefilter": args.pair_prefilter,
        },
        output / "p5_selector.joblib",
    )
    write_json(output / "metadata.json", {
        "manifest": str(manifest_path),
        "supervision_run": args.supervision_run,
        "raw_bank_name": args.raw_bank_name,
        "sira_run_name": args.sira_run_name,
        "cross_fit_folds": folds,
        "candidate_features": candidate_features,
        "pair_features": PAIR_FEATURES,
        "target": "P5.1 soft oracle actions plus same-GT@0.3 pair frequency",
        "evaluation": "cross-fitted predictions with cardinality-first one-to-one pooled PQ",
        "mask_output": "direct selected raw/refined instance RLEs; graph decoder replaces pixel-overlap resolver",
    })
    print("\nP5.2 diagnostics:", json.dumps(diagnostics, indent=2), flush=True)
    print("\nP5.2 top configurations:", flush=True)
    print(ranking.head(20).to_string(index=False), flush=True)
    print("\nBest configuration by fold:", flush=True)
    print(pd.DataFrame(fold_rows).to_string(index=False), flush=True)
    print(f"Best P5.2 configuration: {output / 'best_p5_selector.json'}", flush=True)
    print(f"P5.2 model bundle: {output / 'p5_selector.joblib'}", flush=True)


if __name__ == "__main__":
    main()
