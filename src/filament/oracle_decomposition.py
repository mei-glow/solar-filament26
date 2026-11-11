from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import write_json
from .oof_bank import OOFProposalBank
from .postprocess import mask_to_rle_object, polygon_to_rle, resolve_instances_scored, rle_from_json


@dataclass(frozen=True)
class Variant:
    candidate_index: int
    kind: str
    rle: dict
    area: float


def _one_to_one_stats(iou: np.ndarray, prediction_count: int, gt_count: int) -> dict[str, float | int]:
    """Cardinality-first one-to-one matching for candidate pools that may overlap.

    The fast project PQ helper counts every IoU>0.5 hit, which is equivalent to
    one-to-one matching only after predictions have been made non-overlapping.
    Oracle pools deliberately contain overlapping alternatives, so they require
    explicit bipartite matching to avoid counting multiple predictions for one GT.
    """
    if prediction_count and gt_count:
        valid = iou > 0.5
        # A valid edge bonus larger than any possible sum-IoU difference first
        # maximizes match cardinality, then maximizes IoU among those matchings.
        bonus = min(prediction_count, gt_count) + 1.0
        rows, columns = linear_sum_assignment(-(valid.astype(np.float64) * bonus + iou))
        accepted = valid[rows, columns]
        tp = int(accepted.sum())
        iou_sum = float(iou[rows[accepted], columns[accepted]].sum())
    else:
        tp, iou_sum = 0, 0.0
    return {"iou_sum": iou_sum, "tp": tp, "fp": prediction_count - tp, "fn": gt_count - tp}


def _aggregate(gt_sets: list[list[dict]], predictions: list[dict]) -> dict[str, float | int]:
    output = dict(iou_sum=0.0, tp=0, fp=0, fn=0, pq_sum=0.0, records=0)
    for gt in gt_sets:
        iou = mask_utils.iou(predictions, gt, [0] * len(gt)) if predictions and gt else np.zeros((len(predictions), len(gt)), np.float32)
        values = _one_to_one_stats(iou, len(predictions), len(gt))
        denominator = values["tp"] + 0.5 * values["fp"] + 0.5 * values["fn"]
        for key in ("iou_sum", "tp", "fp", "fn"):
            output[key] += values[key]
        output["pq_sum"] += values["iou_sum"] / denominator if denominator else 0.0
        output["records"] += 1
    denominator = output["tp"] + 0.5 * output["fp"] + 0.5 * output["fn"]
    output["pq"] = output["iou_sum"] / denominator if denominator else 0.0
    output["sq"] = output["iou_sum"] / output["tp"] if output["tp"] else 0.0
    output["rq"] = output["tp"] / denominator if denominator else 0.0
    return output


def _global(per_image: list[dict]) -> dict[str, float | int]:
    iou_sum = float(sum(float(row["iou_sum"]) for row in per_image))
    tp = int(sum(int(row["tp"]) for row in per_image))
    fp = int(sum(int(row["fp"]) for row in per_image))
    fn = int(sum(int(row["fn"]) for row in per_image))
    records = int(sum(int(row["records"]) for row in per_image))
    pq_sum = float(sum(float(row["pq_sum"]) for row in per_image))
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "global_pq": iou_sum / denominator if denominator else 0.0,
        "mean_pq": pq_sum / records if records else 0.0,
        "sq": iou_sum / tp if tp else 0.0,
        "rq": tp / denominator if denominator else 0.0,
        "iou_sum": iou_sum, "tp": tp, "fp": fp, "fn": fn, "annotation_records": records,
    }


def _decode(rles: list[dict], shape: tuple[int, int]) -> np.ndarray:
    if not rles:
        return np.empty((0, *shape), bool)
    decoded = mask_utils.decode(rles)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    return np.moveaxis(decoded, -1, 0).astype(bool, copy=False)


def _resolve(rles: list[dict], scores: np.ndarray, shape: tuple[int, int], threshold: float, min_area: int, duplicate_iou: float) -> list[dict]:
    masks = _decode(rles, shape)
    selected = resolve_instances_scored(masks, scores, threshold, min_area, duplicate_iou)
    return [mask_to_rle_object(mask) for mask, _ in selected]


def _paired_anchor_choice(raw_rles: list[dict], refined_rles: list[dict], gt_sets: list[list[dict]]) -> tuple[list[dict], np.ndarray]:
    raw_total = np.zeros(len(raw_rles), np.float32)
    refined_total = np.zeros(len(raw_rles), np.float32)
    for gt in gt_sets:
        if not gt or not raw_rles:
            continue
        raw_iou = mask_utils.iou(raw_rles, gt, [0] * len(gt))
        refined_iou = mask_utils.iou(refined_rles, gt, [0] * len(gt))
        anchors = raw_iou.argmax(axis=1)
        indices = np.arange(len(raw_rles))
        raw_total += raw_iou[indices, anchors]
        refined_total += refined_iou[indices, anchors]
    use_refined = refined_total > raw_total
    return [refined_rles[i] if use_refined[i] else raw_rles[i] for i in range(len(raw_rles))], use_refined


def _individual_oracle_score(rle: dict, gt_sets: list[list[dict]]) -> float:
    values = []
    for gt in gt_sets:
        if gt:
            values.append(float(mask_utils.iou([rle], gt, [0] * len(gt)).max()))
        else:
            values.append(0.0)
    return float(np.mean(values)) if values else 0.0


def _oracle_subset(
    variants: list[Variant], gt_sets: list[list[dict]], max_predictions: int,
    swap_passes: int = 2,
) -> tuple[list[int], dict]:
    """Greedy + backward removal + replacement under one-variant-per-candidate constraint."""
    cache: dict[tuple[int, ...], dict] = {}
    all_rles = [variant.rle for variant in variants]
    iou_matrices = [
        mask_utils.iou(all_rles, gt, [0] * len(gt)) if all_rles and gt else np.zeros((len(all_rles), len(gt)), np.float32)
        for gt in gt_sets
    ]

    def evaluate(indices: list[int]) -> dict:
        key = tuple(sorted(indices))
        if key not in cache:
            values = dict(iou_sum=0.0, tp=0, fp=0, fn=0, pq_sum=0.0, records=0)
            for gt, matrix in zip(gt_sets, iou_matrices):
                selected = matrix[list(key), :] if key else np.zeros((0, len(gt)), np.float32)
                current = _one_to_one_stats(selected, len(key), len(gt))
                denominator = current["tp"] + 0.5 * current["fp"] + 0.5 * current["fn"]
                for name in ("iou_sum", "tp", "fp", "fn"):
                    values[name] += current[name]
                values["pq_sum"] += current["iou_sum"] / denominator if denominator else 0.0
                values["records"] += 1
            denominator = values["tp"] + 0.5 * values["fp"] + 0.5 * values["fn"]
            values["pq"] = values["iou_sum"] / denominator if denominator else 0.0
            values["sq"] = values["iou_sum"] / values["tp"] if values["tp"] else 0.0
            values["rq"] = values["tp"] / denominator if denominator else 0.0
            cache[key] = values
        return cache[key]

    selected: list[int] = []
    selected_groups: set[int] = set()
    current = evaluate(selected)
    while len(selected) < max_predictions:
        best_index, best_values = None, current
        for index, variant in enumerate(variants):
            if index in selected or variant.candidate_index in selected_groups:
                continue
            values = evaluate(selected + [index])
            if values["pq"] > best_values["pq"] + 1e-12:
                best_index, best_values = index, values
        if best_index is None:
            break
        selected.append(best_index); selected_groups.add(variants[best_index].candidate_index); current = best_values

    changed = True
    while changed and selected:
        changed = False
        for index in list(selected):
            trial = [value for value in selected if value != index]
            values = evaluate(trial)
            if values["pq"] > current["pq"] + 1e-12:
                selected = trial; selected_groups = {variants[value].candidate_index for value in selected}
                current = values; changed = True; break

    for _ in range(swap_passes):
        best_swap, best_values = None, current
        for old in selected:
            base = [value for value in selected if value != old]
            base_groups = {variants[value].candidate_index for value in base}
            for new, variant in enumerate(variants):
                if new in selected or variant.candidate_index in base_groups:
                    continue
                values = evaluate(base + [new])
                if values["pq"] > best_values["pq"] + 1e-12:
                    best_swap, best_values = (old, new), values
        if best_swap is None:
            break
        old, new = best_swap
        selected = [new if value == old else value for value in selected]; current = best_values
    return sorted(selected), current


def _audit_gt(gt_sets: list[list[dict]], raw_rles: list[dict], final_rles: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for gt in gt_sets:
        if raw_rles and gt:
            raw_best = mask_utils.iou(raw_rles, gt, [0] * len(gt)).max(axis=0)
        else:
            raw_best = np.zeros(len(gt), np.float32)
        if final_rles and gt:
            final_best = mask_utils.iou(final_rles, gt, [0] * len(gt)).max(axis=0)
        else:
            final_best = np.zeros(len(gt), np.float32)
        for raw_iou, final_iou in zip(raw_best, final_best):
            if final_iou > 0.5:
                counts["final_matched"] += 1
            elif raw_iou <= 0.1:
                counts["true_proposal_miss_at_01"] += 1
            elif raw_iou > 0.5:
                counts["selection_or_refinement_induced_miss"] += 1
            else:
                counts["weak_proposal_or_mask_iou_failure"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="O1-O5 oracle decomposition for the M2 OOF candidate pool")
    add_path_args(parser)
    parser.add_argument("--run-name", default="oracle_o1_o5_m2")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--raw-bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--sira-run-name", default="m2_sira_oof")
    parser.add_argument("--quality-run", default="m4_quality_m2")
    parser.add_argument("--oracle-min-area", type=int, default=25)
    parser.add_argument("--resolver-min-area", type=int, default=None)
    parser.add_argument("--duplicate-iou", type=float, default=0.80)
    parser.add_argument("--max-extra-predictions", type=int, default=5)
    parser.add_argument("--swap-passes", type=int, default=2)
    args = parser.parse_args()

    paths = paths_from_args(args); annotation_path, _, _ = discover_data(paths)
    manifest_path = (args.manifest or paths.results_root / "fold_manifest_s3020.csv").resolve()
    manifest = pd.read_csv(manifest_path)
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]: annotations[str(item["image_id"])].append(item)
    quality_dir = paths.results_root / args.quality_run
    quality_config = json.loads((quality_dir / "best_quality.json").read_text())
    quality_threshold = float(quality_config["quality_threshold"])
    resolver_min_area = int(args.resolver_min_area or quality_config["min_area"])
    quality_frame = pd.read_csv(quality_dir / "oof_quality_predictions.csv")
    quality_lookup = {
        (str(row.file_name), int(row.candidate_index)): float(row.quality_probability)
        for row in quality_frame.itertuples(index=False)
    }

    stages = {name: [] for name in ("O1_current_m4", "O2_paired_choice_current_selection", "O3_refined_oracle_set", "O4_raw_refined_oracle_set", "O5_o4_after_resolver")}
    per_image_rows, selection_rows = [], []
    audit = defaultdict(int)
    grouped = list(manifest.groupby("file_name", sort=True))
    for file_name, records in tqdm(grouped, desc="O1-O5 oracle decomposition", unit="image"):
        fold = int(records["fold"].iloc[0]); first_info = images[str(records.iloc[0]["image_id"])]
        shape = (int(first_info["height"]), int(first_info["width"]))
        gt_sets = []
        for record in records.itertuples(index=False):
            image_id = str(record.image_id); info = images[image_id]
            gt_sets.append([polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]])
        raw_payload = OOFProposalBank(paths.artifact_root / "predictions" / "oof" / args.raw_bank_name / f"fold{fold}").load(file_name, decode_masks=False)
        refined_payload = OOFProposalBank(paths.artifact_root / "predictions" / "sira_oof" / args.sira_run_name / f"fold{fold}").load(file_name, decode_masks=False)
        raw_rles = [rle_from_json(item["rle"]) for item in raw_payload["instances"]]
        refined_rles = [rle_from_json(item["rle"]) for item in refined_payload["instances"]]
        if len(raw_rles) != len(refined_rles):
            raise RuntimeError(f"Raw/refined mismatch for {file_name}")
        quality_scores = np.asarray([quality_lookup[(file_name, index)] for index in range(len(refined_rles))], np.float32)

        o1_predictions = _resolve(refined_rles, quality_scores, shape, quality_threshold, resolver_min_area, args.duplicate_iou)
        paired_rles, paired_use_refined = _paired_anchor_choice(raw_rles, refined_rles, gt_sets)
        o2_predictions = _resolve(paired_rles, quality_scores, shape, quality_threshold, resolver_min_area, args.duplicate_iou)

        refined_variants = [
            Variant(index, "refined", rle, float(mask_utils.area(rle)))
            for index, rle in enumerate(refined_rles) if float(mask_utils.area(rle)) >= args.oracle_min_area
        ]
        combined_variants = []
        for index, (raw_rle, refined_rle) in enumerate(zip(raw_rles, refined_rles)):
            raw_area, refined_area = float(mask_utils.area(raw_rle)), float(mask_utils.area(refined_rle))
            if raw_area >= args.oracle_min_area: combined_variants.append(Variant(index, "raw", raw_rle, raw_area))
            if refined_area >= args.oracle_min_area: combined_variants.append(Variant(index, "refined", refined_rle, refined_area))
        max_predictions = max((len(gt) for gt in gt_sets), default=0) + args.max_extra_predictions
        o3_indices, _ = _oracle_subset(refined_variants, gt_sets, max_predictions, args.swap_passes)
        o4_indices, _ = _oracle_subset(combined_variants, gt_sets, max_predictions, args.swap_passes)
        o3_predictions = [refined_variants[index].rle for index in o3_indices]
        o4_predictions = [combined_variants[index].rle for index in o4_indices]
        o4_scores = np.asarray([_individual_oracle_score(combined_variants[index].rle, gt_sets) for index in o4_indices], np.float32)
        o5_predictions = _resolve(o4_predictions, o4_scores, shape, 0.0, resolver_min_area, args.duplicate_iou)

        predictions_by_stage = {
            "O1_current_m4": o1_predictions,
            "O2_paired_choice_current_selection": o2_predictions,
            "O3_refined_oracle_set": o3_predictions,
            "O4_raw_refined_oracle_set": o4_predictions,
            "O5_o4_after_resolver": o5_predictions,
        }
        row = {"file_name": file_name, "fold": fold, "annotation_records": len(gt_sets), "raw_candidates": len(raw_rles)}
        for stage, predictions in predictions_by_stage.items():
            values = _aggregate(gt_sets, predictions); stages[stage].append(values)
            row[f"{stage}_pq"] = values["pq"]; row[f"{stage}_predictions"] = len(predictions)
        per_image_rows.append(row)
        for key, value in _audit_gt(gt_sets, raw_rles, o1_predictions).items(): audit[key] += value
        selection_rows.append({
            "file_name": file_name, "fold": fold,
            "paired_refined_fraction": float(paired_use_refined.mean()) if len(paired_use_refined) else 0.0,
            "o3_selected": len(o3_indices), "o4_selected": len(o4_indices),
            "o4_raw": sum(combined_variants[index].kind == "raw" for index in o4_indices),
            "o4_refined": sum(combined_variants[index].kind == "refined" for index in o4_indices),
            "o5_selected": len(o5_predictions),
        })

    summary = {stage: _global(values) for stage, values in stages.items()}
    summary["audit"] = dict(audit)
    summary["configuration"] = {
        "quality_threshold": quality_threshold, "resolver_min_area": resolver_min_area,
        "oracle_min_area": args.oracle_min_area, "duplicate_iou": args.duplicate_iou,
        "max_extra_predictions": args.max_extra_predictions, "swap_passes": args.swap_passes,
        "oracle_note": "O3/O4 are greedy+removal+swap empirical ceilings, not guaranteed global optima.",
    }
    output = paths.experiment_results(args.run_name)
    write_json(output / "oracle_decomposition.json", summary)
    pd.DataFrame(per_image_rows).to_csv(output / "oracle_by_image.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "oracle_selection_details.csv", index=False)
    table = pd.DataFrame([
        {"stage": stage, **{key: values[key] for key in ("global_pq", "mean_pq", "sq", "rq", "tp", "fp", "fn")}}
        for stage, values in summary.items() if stage.startswith("O")
    ])
    table.to_csv(output / "oracle_summary.csv", index=False)
    print(table.to_string(index=False), flush=True)
    print("\nMiss audit:", json.dumps(dict(audit), indent=2), flush=True)
    print(f"Saved: {output / 'oracle_decomposition.json'}", flush=True)


if __name__ == "__main__":
    main()
