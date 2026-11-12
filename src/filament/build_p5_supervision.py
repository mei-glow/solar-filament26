from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import write_json
from .gated_selection import FEATURES, build_candidate_features
from .oof_bank import OOFProposalBank
from .oracle_decomposition import Variant, _aggregate, _oracle_subset
from .postprocess import polygon_to_rle, rle_from_json


ACTIONS = ("reject", "raw", "refined")


def _matching_pairs(iou: np.ndarray) -> list[tuple[int, int, float]]:
    prediction_count, gt_count = iou.shape
    if not prediction_count or not gt_count:
        return []
    valid = iou > 0.5
    bonus = min(prediction_count, gt_count) + 1.0
    rows, columns = linear_sum_assignment(-(valid.astype(np.float64) * bonus + iou))
    return [(int(row), int(column), float(iou[row, column])) for row, column in zip(rows, columns) if valid[row, column]]


def _soft_actions(utilities: dict[str, float], temperature: float) -> dict[str, float]:
    available = {key: value for key, value in utilities.items() if np.isfinite(value)}
    maximum = max(available.values())
    weights = {key: float(np.exp((value - maximum) / temperature)) for key, value in available.items()}
    denominator = sum(weights.values())
    return {action: weights.get(action, 0.0) / denominator for action in ACTIONS}


def _paired_box_iou(boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 0), np.float32)
    x, y, width, height = boxes.T; x2, y2 = x + width, y + height
    inter_width = np.maximum(0.0, np.minimum(x2[:, None], x2[None]) - np.maximum(x[:, None], x[None]))
    inter_height = np.maximum(0.0, np.minimum(y2[:, None], y2[None]) - np.maximum(y[:, None], y[None]))
    intersection = inter_width * inter_height
    area = width * height
    return intersection / np.maximum(area[:, None] + area[None] - intersection, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.1: export OOF oracle supervision for set-aware candidate selection")
    add_path_args(parser)
    parser.add_argument("--run-name", default="p5_1_oracle_supervision_m2")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw-bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--sira-run-name", default="m2_sira_oof")
    parser.add_argument("--oracle-min-area", type=int, default=25)
    parser.add_argument("--max-extra-predictions", type=int, default=5)
    parser.add_argument("--swap-passes", type=int, default=2)
    parser.add_argument("--action-temperature", type=float, default=0.01)
    args = parser.parse_args()
    if args.action_temperature <= 0:
        parser.error("--action-temperature must be positive")

    paths = paths_from_args(args); annotation_path, _, _ = discover_data(paths)
    manifest_path = (args.manifest or paths.results_root / "fold_manifest_s3020.csv").resolve()
    manifest = pd.read_csv(manifest_path)
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]: annotations[str(item["image_id"])].append(item)

    candidate_rows, annotation_rows, pair_rows, image_rows = [], [], [], []
    grouped = list(manifest.groupby("file_name", sort=True))
    for file_name, records in tqdm(grouped, desc="P5.1 oracle supervision", unit="image"):
        fold = int(records["fold"].iloc[0])
        raw_payload = OOFProposalBank(paths.artifact_root / "predictions" / "oof" / args.raw_bank_name / f"fold{fold}").load(file_name, decode_masks=False)
        refined_payload = OOFProposalBank(paths.artifact_root / "predictions" / "sira_oof" / args.sira_run_name / f"fold{fold}").load(file_name, decode_masks=False)
        raw_instances, refined_instances = raw_payload["instances"], refined_payload["instances"]
        if len(raw_instances) != len(refined_instances):
            raise RuntimeError(f"Raw/refined mismatch for {file_name}")
        count = len(raw_instances)
        features = build_candidate_features(raw_instances, refined_instances)
        raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
        refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
        raw_area = np.asarray(mask_utils.area(raw_rles), np.float32).reshape(-1) if count else np.empty(0, np.float32)
        refined_area = np.asarray(mask_utils.area(refined_rles), np.float32).reshape(-1) if count else np.empty(0, np.float32)
        gt_sets, annotation_ids = [], []
        for record in records.itertuples(index=False):
            image_id = str(record.image_id); info = images[image_id]
            items = annotations[image_id]
            gt_sets.append([polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in items])
            annotation_ids.append([str(item["id"]) for item in items])

        variants, variant_lookup = [], {}
        for index, (raw_rle, refined_rle) in enumerate(zip(raw_rles, refined_rles)):
            if raw_area[index] >= args.oracle_min_area:
                variant_lookup[(index, "raw")] = len(variants); variants.append(Variant(index, "raw", raw_rle, float(raw_area[index])))
            if refined_area[index] >= args.oracle_min_area:
                variant_lookup[(index, "refined")] = len(variants); variants.append(Variant(index, "refined", refined_rle, float(refined_area[index])))
        max_predictions = max((len(gt) for gt in gt_sets), default=0) + args.max_extra_predictions
        selected_indices, base_values = _oracle_subset(variants, gt_sets, max_predictions, args.swap_passes)
        selected_by_candidate = {variants[index].candidate_index: index for index in selected_indices}
        selected_rles = [variants[index].rle for index in selected_indices]

        raw_anchor_sum = np.zeros(count, np.float64); refined_anchor_sum = np.zeros(count, np.float64)
        raw_best_sum = np.zeros(count, np.float64); refined_best_sum = np.zeros(count, np.float64)
        raw_pass = np.zeros(count, np.float64); refined_pass = np.zeros(count, np.float64)
        gain_cross = np.zeros(count, np.float64); damage_cross = np.zeros(count, np.float64)
        identity_switch = np.zeros(count, np.float64); oracle_match = np.zeros(count, np.float64)
        same01 = np.zeros((count, count), np.float32); same03 = np.zeros((count, count), np.float32); same05 = np.zeros((count, count), np.float32)

        selected_positions = {variant_index: position for position, variant_index in enumerate(selected_indices)}
        for record, gt, gt_ids in zip(records.itertuples(index=False), gt_sets, annotation_ids):
            raw_iou = mask_utils.iou(raw_rles, gt, [0] * len(gt)) if raw_rles and gt else np.zeros((count, len(gt)), np.float32)
            refined_iou = mask_utils.iou(refined_rles, gt, [0] * len(gt)) if refined_rles and gt else np.zeros((count, len(gt)), np.float32)
            if len(gt):
                raw_anchor = raw_iou.argmax(axis=1); refined_anchor = refined_iou.argmax(axis=1)
                indices = np.arange(count)
                raw_anchor_values = raw_iou[indices, raw_anchor]; refined_same_anchor = refined_iou[indices, raw_anchor]
                raw_best = raw_anchor_values; refined_best = refined_iou[indices, refined_anchor]
                identity_quality = np.maximum(raw_best, refined_best)
                identity_anchor = np.where(refined_best > raw_best, refined_anchor, raw_anchor)
            else:
                raw_anchor = refined_anchor = identity_anchor = np.full(count, -1, int)
                raw_anchor_values = refined_same_anchor = raw_best = refined_best = identity_quality = np.zeros(count, np.float32)

            raw_anchor_sum += raw_anchor_values; refined_anchor_sum += refined_same_anchor
            raw_best_sum += raw_best; refined_best_sum += refined_best
            raw_pass += raw_anchor_values > 0.5; refined_pass += refined_same_anchor > 0.5
            gain_cross += (raw_anchor_values <= 0.5) & (refined_same_anchor > 0.5)
            damage_cross += (raw_anchor_values > 0.5) & (refined_same_anchor <= 0.5)
            identity_switch += (raw_anchor != refined_anchor) & (raw_best > 0.1) & (refined_best > 0.1)

            for threshold, accumulator in ((0.1, same01), (0.3, same03), (0.5, same05)):
                valid = identity_quality > threshold
                accumulator += (identity_anchor[:, None] == identity_anchor[None, :]) & valid[:, None] & valid[None, :]

            selected_iou = mask_utils.iou(selected_rles, gt, [0] * len(gt)) if selected_rles and gt else np.zeros((len(selected_rles), len(gt)), np.float32)
            pairs = _matching_pairs(selected_iou)
            matched_selected = {prediction_index: (gt_index, iou) for prediction_index, gt_index, iou in pairs}
            for selected_position, variant_index in enumerate(selected_indices):
                candidate_index = variants[variant_index].candidate_index
                if selected_position in matched_selected:
                    oracle_match[candidate_index] += 1

            for index in range(count):
                anchor_id = gt_ids[int(raw_anchor[index])] if len(gt) else ""
                refined_best_id = gt_ids[int(refined_anchor[index])] if len(gt) else ""
                selected_variant_index = selected_by_candidate.get(index)
                selected_kind = variants[selected_variant_index].kind if selected_variant_index is not None else "reject"
                match_gt_id, match_iou = "", 0.0
                if selected_variant_index is not None:
                    selected_position = selected_positions[selected_variant_index]
                    if selected_position in matched_selected:
                        gt_index, match_iou = matched_selected[selected_position]; match_gt_id = gt_ids[gt_index]
                annotation_rows.append({
                    "file_name": file_name, "fold": fold, "image_id": str(record.image_id), "candidate_index": index,
                    "raw_anchor_annotation_id": anchor_id, "refined_best_annotation_id": refined_best_id,
                    "raw_anchor_iou": float(raw_anchor_values[index]), "refined_same_anchor_iou": float(refined_same_anchor[index]),
                    "raw_best_iou": float(raw_best[index]), "refined_best_iou": float(refined_best[index]),
                    "delta_same_anchor_iou": float(refined_same_anchor[index] - raw_anchor_values[index]),
                    "threshold_crossing": int(refined_same_anchor[index] > 0.5) - int(raw_anchor_values[index] > 0.5),
                    "identity_switch": bool(raw_anchor[index] != refined_anchor[index] and raw_best[index] > 0.1 and refined_best[index] > 0.1),
                    "oracle_action": selected_kind, "oracle_matched_annotation_id": match_gt_id, "oracle_matched_iou": float(match_iou),
                })

        record_count = max(len(gt_sets), 1)
        base_set = list(selected_indices)
        base_pq = float(base_values["pq"])
        action_probabilities, action_utilities = {}, {}
        for candidate_index in range(count):
            without = [index for index in base_set if variants[index].candidate_index != candidate_index]
            utilities = {"reject": float(_aggregate(gt_sets, [variants[index].rle for index in without])["pq"]), "raw": -np.inf, "refined": -np.inf}
            for action in ("raw", "refined"):
                variant_index = variant_lookup.get((candidate_index, action))
                if variant_index is not None:
                    utilities[action] = float(_aggregate(gt_sets, [variants[index].rle for index in without + [variant_index]])["pq"])
            probabilities = _soft_actions(utilities, args.action_temperature)
            action_utilities[candidate_index] = utilities; action_probabilities[candidate_index] = probabilities

        for index in range(count):
            selected_variant_index = selected_by_candidate.get(index)
            oracle_action = variants[selected_variant_index].kind if selected_variant_index is not None else "reject"
            utilities, probabilities = action_utilities[index], action_probabilities[index]
            finite_values = sorted((value for value in utilities.values() if np.isfinite(value)), reverse=True)
            feature_row = features.iloc[index]
            candidate_rows.append({
                "file_name": file_name, "fold": fold, "candidate_index": index,
                **{name: float(feature_row[name]) for name in FEATURES},
                "raw_anchor_iou_mean": float(raw_anchor_sum[index] / record_count),
                "refined_same_anchor_iou_mean": float(refined_anchor_sum[index] / record_count),
                "delta_same_anchor_iou_mean": float((refined_anchor_sum[index] - raw_anchor_sum[index]) / record_count),
                "raw_best_iou_mean": float(raw_best_sum[index] / record_count),
                "refined_best_iou_mean": float(refined_best_sum[index] / record_count),
                "raw_pass_frequency": float(raw_pass[index] / record_count),
                "refined_pass_frequency": float(refined_pass[index] / record_count),
                "gain_crossing_frequency": float(gain_cross[index] / record_count),
                "damage_crossing_frequency": float(damage_cross[index] / record_count),
                "identity_switch_frequency": float(identity_switch[index] / record_count),
                "oracle_action": oracle_action, "oracle_selected": oracle_action != "reject",
                "oracle_match_frequency": float(oracle_match[index] / record_count),
                "oracle_base_image_pq": base_pq,
                "oracle_reject_pq": utilities["reject"], "oracle_raw_pq": utilities["raw"], "oracle_refined_pq": utilities["refined"],
                "oracle_action_margin": float(finite_values[0] - finite_values[1]) if len(finite_values) > 1 else 0.0,
                "target_p_reject": probabilities["reject"], "target_p_raw": probabilities["raw"], "target_p_refined": probabilities["refined"],
            })

        if count:
            raw_mask_iou = mask_utils.iou(raw_rles, raw_rles, [0] * count)
            refined_mask_iou = mask_utils.iou(refined_rles, refined_rles, [0] * count)
            raw_refined_cross = mask_utils.iou(raw_rles, refined_rles, [0] * count)
            raw_boxes = np.asarray(mask_utils.toBbox(raw_rles), np.float32).reshape(-1, 4)
            box_iou = _paired_box_iou(raw_boxes)
            centers = raw_boxes[:, :2] + raw_boxes[:, 2:] / 2.0
            image_info = images[str(records.iloc[0]["image_id"])]
            diagonal = float(np.hypot(image_info["height"], image_info["width"]))
            center_distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=2) / max(diagonal, 1.0)
            detector_scores = features["detector_score"].to_numpy(np.float32)
            semantic_support = features["semantic_support"].to_numpy(np.float32)
            selected_groups = set(selected_by_candidate)
            for left in range(count):
                for right in range(left + 1, count):
                    pair_rows.append({
                        "file_name": file_name, "fold": fold, "left_candidate_index": left, "right_candidate_index": right,
                        "raw_mask_iou": float(raw_mask_iou[left, right]), "refined_mask_iou": float(refined_mask_iou[left, right]),
                        "max_cross_variant_iou": float(max(raw_refined_cross[left, right], raw_refined_cross[right, left], raw_mask_iou[left, right], refined_mask_iou[left, right])),
                        "raw_box_iou": float(box_iou[left, right]), "normalized_center_distance": float(center_distance[left, right]),
                        "absolute_log_area_ratio": float(abs(np.log((raw_area[left] + 1.0) / (raw_area[right] + 1.0)))),
                        "detector_score_difference": float(abs(detector_scores[left] - detector_scores[right])),
                        "semantic_support_difference": float(abs(semantic_support[left] - semantic_support[right])),
                        "same_gt_frequency_01": float(same01[left, right] / record_count),
                        "same_gt_frequency_03": float(same03[left, right] / record_count),
                        "same_gt_frequency_05": float(same05[left, right] / record_count),
                        "oracle_both_selected": left in selected_groups and right in selected_groups,
                    })
        image_rows.append({
            "file_name": file_name, "fold": fold, "annotation_records": len(gt_sets), "candidates": count,
            "oracle_pq": base_pq, "oracle_selected": len(selected_indices),
            "oracle_raw": sum(variants[index].kind == "raw" for index in selected_indices),
            "oracle_refined": sum(variants[index].kind == "refined" for index in selected_indices),
            "oracle_iou_sum": float(base_values["iou_sum"]), "oracle_tp": int(base_values["tp"]),
            "oracle_fp": int(base_values["fp"]), "oracle_fn": int(base_values["fn"]),
        })

    output = paths.experiment_results(args.run_name)
    candidates = pd.DataFrame(candidate_rows); pairs = pd.DataFrame(pair_rows); annotations_frame = pd.DataFrame(annotation_rows); images_frame = pd.DataFrame(image_rows)
    candidates.to_csv(output / "candidate_supervision.csv", index=False)
    annotations_frame.to_csv(output / "candidate_annotation_targets.csv.gz", index=False, compression="gzip")
    pairs.to_csv(output / "pair_supervision.csv.gz", index=False, compression="gzip")
    images_frame.to_csv(output / "image_oracle_summary.csv", index=False)
    action_counts = candidates["oracle_action"].value_counts().to_dict() if len(candidates) else {}
    pooled_iou = float(images_frame["oracle_iou_sum"].sum())
    pooled_tp = int(images_frame["oracle_tp"].sum())
    pooled_fp = int(images_frame["oracle_fp"].sum())
    pooled_fn = int(images_frame["oracle_fn"].sum())
    pooled_denominator = pooled_tp + 0.5 * pooled_fp + 0.5 * pooled_fn
    summary = {
        "physical_images": int(images_frame["file_name"].nunique()), "candidates": len(candidates),
        "annotation_candidate_rows": len(annotations_frame), "pairs": len(pairs),
        "oracle_action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "oracle_global_pq": pooled_iou / pooled_denominator if pooled_denominator else 0.0,
        "oracle_tp": pooled_tp, "oracle_fp": pooled_fp, "oracle_fn": pooled_fn,
        "mean_oracle_image_pq": float(images_frame["oracle_pq"].mean()),
        "mean_selected_per_image": float(images_frame["oracle_selected"].mean()),
        "pair_positive_rate_03": float((pairs["same_gt_frequency_03"] > 0).mean()) if len(pairs) else 0.0,
        "configuration": vars(args),
        "feature_columns": FEATURES,
        "target_notes": {
            "paired_gate": "raw and refined IoU use the same raw-anchored GT within each annotation record",
            "oracle_action": "one action per proposal from corrected O4 greedy+removal+swap selection",
            "soft_action": "local reject/raw/refined replacement utilities converted with temperature softmax",
            "duplicate": "same-GT frequency uses the stronger raw/refined identity anchor and multiple IoU thresholds",
        },
    }
    write_json(output / "supervision_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"configuration", "feature_columns", "target_notes"}}, indent=2), flush=True)
    print(f"Candidate supervision: {output / 'candidate_supervision.csv'}", flush=True)
    print(f"Pair supervision: {output / 'pair_supervision.csv.gz'}", flush=True)


if __name__ == "__main__":
    main()
