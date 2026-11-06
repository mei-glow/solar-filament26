from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from tqdm import tqdm

from .analysis.error_taxonomy import classify_record
from .config import add_path_args, discover_data, paths_from_args
from .data.annotators import select_annotation_records
from .experiment import write_json
from .oof_bank import OOFProposalBank
from .postprocess import (
    mask_to_rle_object,
    polygon_to_rle,
    pq_stats_rle,
    resolve_instances_scored,
    rle_from_json,
)


def _load_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("Diagnostics require matplotlib. Install it with: pip install matplotlib") from error
    return plt


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _decode_many(rles: list[dict], shape: tuple[int, int]) -> np.ndarray:
    if not rles:
        return np.empty((0, *shape), dtype=bool)
    decoded = mask_utils.decode(rles)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    return np.moveaxis(decoded, -1, 0).astype(bool, copy=False)


def _geometry(rle: dict, height: int, width: int) -> dict[str, float]:
    x, y, box_width, box_height = map(float, mask_utils.toBbox(rle))
    center_x, center_y = x + box_width / 2.0, y + box_height / 2.0
    radial = np.hypot(center_x - width / 2.0, center_y - height / 2.0) / (min(height, width) / 2.0)
    elongation = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
    return {
        "area": float(mask_utils.area(rle)),
        "radial_position": float(radial),
        "elongation": float(elongation),
    }


def _best_iou(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros(len(gt_rles), dtype=np.float32)
    return mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).max(axis=0)


def _candidate_iou(pred_rles: list[dict], gt_rles: list[dict]) -> np.ndarray:
    if not pred_rles or not gt_rles:
        return np.zeros(len(pred_rles), dtype=np.float32)
    return mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).max(axis=1)


def _aggregate_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    tp, fp, fn = int(frame["tp"].sum()), int(frame["fp"].sum()), int(frame["fn"].sum())
    iou_sum = float(frame["iou_sum"].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "global_pq": _safe_div(iou_sum, denominator),
        "mean_pq": float(frame["pq"].mean()),
        "sq": _safe_div(iou_sum, tp),
        "rq": _safe_div(tp, denominator),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "iou_sum": iou_sum,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "annotation_records": int(len(frame)),
        "physical_images": int(frame["file_name"].nunique()),
    }


def _stratified(gt_frame: pd.DataFrame, column: str, labels: list[str], bins) -> pd.DataFrame:
    values = gt_frame.copy()
    if bins == "quantile":
        values["bin"] = pd.qcut(values[column].rank(method="first"), len(labels), labels=labels)
    else:
        values["bin"] = pd.cut(values[column], bins=bins, labels=labels, include_lowest=True)
    return values.groupby("bin", observed=True).agg(
        gt_instances=("annotation_id", "size"),
        proposal_recall_01=("raw_iou_01", "mean"),
        proposal_recall_03=("raw_iou_03", "mean"),
        proposal_recall_05=("raw_iou_05", "mean"),
        refined_candidate_recall_05=("refined_iou_05", "mean"),
        final_recall_05=("final_iou_05", "mean"),
        mean_raw_best_iou=("raw_best_iou", "mean"),
        mean_refined_best_iou=("refined_best_iou", "mean"),
        mean_final_best_iou=("final_best_iou", "mean"),
        median_value=(column, "median"),
    ).reset_index()


def _bar_plot(plt, frame: pd.DataFrame, x: str, y: str, title: str, target: Path, ylim=None) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(frame[x].astype(str), frame[y].astype(float), color="#3568a8")
    axis.set_title(title); axis.set_xlabel(""); axis.set_ylabel(y.replace("_", " "))
    if ylim is not None: axis.set_ylim(*ylim)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(target, dpi=160); plt.close(figure)


def _color_instances(shape: tuple[int, int], masks: list[np.ndarray]) -> np.ndarray:
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    palette = [(31, 196, 255), (255, 97, 74), (92, 220, 92), (202, 112, 255), (255, 210, 71)]
    for index, mask in enumerate(masks):
        canvas[mask] = palette[index % len(palette)]
    return canvas


def _overlay(gray: np.ndarray, color: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    active = color.any(axis=2)
    output = rgb.copy()
    output[active] = cv2.addWeighted(rgb, 1.0 - alpha, color, alpha, 0)[active]
    return output


def _title(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _render_case(
    image_path: Path, gt_rles: list[dict], raw_rles: list[dict], final_rles: list[dict],
    raw_scores: np.ndarray, raw_conf: float, min_area: int, pq: float, target: Path,
) -> np.ndarray:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    shape = gray.shape
    gt_masks = _decode_many(gt_rles, shape)
    raw_masks = _decode_many(raw_rles, shape)
    raw_selected = resolve_instances_scored(raw_masks, raw_scores, raw_conf, min_area) if len(raw_masks) else []
    final_masks = _decode_many(final_rles, shape)
    gt_union = gt_masks.any(axis=0) if len(gt_masks) else np.zeros(shape, bool)
    pred_union = final_masks.any(axis=0) if len(final_masks) else np.zeros(shape, bool)
    error = np.zeros((*shape, 3), np.uint8)
    error[gt_union & pred_union] = (40, 210, 40)
    error[gt_union & ~pred_union] = (0, 220, 255)
    error[~gt_union & pred_union] = (30, 30, 235)
    panels = [
        _title(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), image_path.name),
        _title(_overlay(gray, _color_instances(shape, list(gt_masks))), f"GT medoid: {len(gt_masks)}"),
        _title(_overlay(gray, _color_instances(shape, [mask for mask, _ in raw_selected])), f"Raw proposals: {len(raw_selected)}"),
        _title(_overlay(gray, error), f"Final: {len(final_masks)}  PQ={pq:.3f}"),
    ]
    panels = [cv2.resize(panel, (512, 512), interpolation=cv2.INTER_AREA) for panel in panels]
    composite = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), composite, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return cv2.resize(composite, (512, 512), interpolation=cv2.INTER_AREA)


def _gallery(tiles: list[np.ndarray], target: Path, columns: int = 4) -> None:
    if not tiles:
        return
    blank = np.zeros_like(tiles[0]); rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns] + [blank] * max(0, columns - len(tiles[start:start + columns]))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(target), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 91])


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive quantitative and qualitative diagnostics for OOF SIRA")
    add_path_args(parser)
    parser.add_argument("--method", choices=("m2", "m3"), default="m2")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--quality-run", default=None)
    parser.add_argument("--raw-bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--sira-run-name", default=None)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--raw-conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=3020)
    args = parser.parse_args()

    paths = paths_from_args(args)
    annotation_path, image_dir, _ = discover_data(paths)
    manifest_path = (args.manifest or paths.results_root / "fold_manifest_s3020.csv").resolve()
    quality_run = args.quality_run or f"m4_quality_{args.method}"
    sira_run_name = args.sira_run_name or f"{args.method}_sira_oof"
    quality_dir = paths.results_root / quality_run
    output_dir = paths.experiment_results(f"diagnostics_{args.method}")
    charts_dir, qualitative_dir = output_dir / "charts", output_dir / "qualitative"
    charts_dir.mkdir(parents=True, exist_ok=True); qualitative_dir.mkdir(parents=True, exist_ok=True)
    quality_config = json.loads((quality_dir / "best_quality.json").read_text())
    quality_threshold, min_area = float(quality_config["quality_threshold"]), int(quality_config["min_area"])
    quality_predictions = pd.read_csv(quality_dir / "oof_quality_predictions.csv")
    quality_lookup = {
        (str(row.file_name), int(row.candidate_index)): float(row.quality_probability)
        for row in quality_predictions.itertuples(index=False)
    }
    manifest = pd.read_csv(manifest_path)
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]: annotations[str(item["image_id"])].append(item)
    medoid = select_annotation_records(manifest, coco, "medoid", args.seed)
    medoid_by_file = {row.file_name: str(row.image_id) for row in medoid.itertuples(index=False)}

    metric_rows, taxonomy_rows, gt_rows, candidate_rows = [], [], [], []
    final_by_file: dict[str, list[dict]] = {}
    raw_by_file: dict[str, tuple[list[dict], np.ndarray]] = {}
    transition_counts = Counter()

    for file_name, records in tqdm(list(manifest.groupby("file_name", sort=True)), desc=f"Diagnose {args.method.upper()}", unit="image"):
        fold = int(records["fold"].iloc[0])
        raw_payload = OOFProposalBank(paths.artifact_root / "predictions" / "oof" / args.raw_bank_name / f"fold{fold}").load(file_name, decode_masks=False)
        refined_payload = OOFProposalBank(paths.artifact_root / "predictions" / "sira_oof" / sira_run_name / f"fold{fold}").load(file_name, decode_masks=False)
        raw_instances, refined_instances = raw_payload["instances"], refined_payload["instances"]
        raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
        refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
        raw_scores = np.asarray([float(item["score"]) for item in raw_instances], np.float32)
        quality_scores = np.asarray([quality_lookup[(file_name, index)] for index in range(len(refined_instances))], np.float32)
        info = images[str(records.iloc[0]["image_id"])]
        refined_masks = _decode_many(refined_rles, (int(info["height"]), int(info["width"])))
        selected = resolve_instances_scored(refined_masks, quality_scores, quality_threshold, min_area)
        final_rles = [mask_to_rle_object(mask) for mask, _ in selected]
        final_by_file[file_name] = final_rles
        raw_by_file[file_name] = (raw_rles, raw_scores)

        medoid_id = medoid_by_file[file_name]
        medoid_info = images[medoid_id]
        medoid_gt = [polygon_to_rle(item["segmentation"], medoid_info["height"], medoid_info["width"]) for item in annotations[medoid_id]]
        raw_candidate_iou = _candidate_iou(raw_rles, medoid_gt)
        refined_candidate_iou = _candidate_iou(refined_rles, medoid_gt)
        for index in range(min(len(raw_instances), len(refined_instances))):
            before, after = float(raw_candidate_iou[index]), float(refined_candidate_iou[index])
            if before > 0.5 and after > 0.5: transition = "preserved_tp_candidate"
            elif before <= 0.5 and after > 0.5: transition = "rescued_over_05"
            elif before > 0.5 and after <= 0.5: transition = "damaged_below_05"
            else: transition = "still_below_05"
            transition_counts[transition] += 1
            candidate_rows.append({
                "file_name": file_name, "fold": fold, "candidate_index": index,
                "detector_score": float(raw_instances[index]["score"]),
                "quality_probability": float(quality_scores[index]),
                "raw_best_iou": before, "refined_best_iou": after,
                "iou_delta": after - before, "transition": transition,
                "quality_pass": bool(quality_scores[index] >= quality_threshold),
            })

        for record in records.itertuples(index=False):
            image_id = str(record.image_id); record_info = images[image_id]
            gt_rles = [polygon_to_rle(item["segmentation"], record_info["height"], record_info["width"]) for item in annotations[image_id]]
            values = pq_stats_rle(gt_rles, final_rles)
            metric_rows.append({"file_name": file_name, "image_id": image_id, "fold": fold, "gt": len(gt_rles), "pred": len(final_rles), **values})
            taxonomy_rows.append({"file_name": file_name, "image_id": image_id, "fold": fold, **classify_record(gt_rles, final_rles)})
            raw_best = _best_iou(gt_rles, raw_rles); refined_best = _best_iou(gt_rles, refined_rles); final_best = _best_iou(gt_rles, final_rles)
            for index, (annotation, gt_rle) in enumerate(zip(annotations[image_id], gt_rles)):
                geometry = _geometry(gt_rle, int(record_info["height"]), int(record_info["width"]))
                gt_rows.append({
                    "file_name": file_name, "image_id": image_id, "annotation_id": annotation["id"], "fold": fold,
                    **geometry,
                    "raw_best_iou": float(raw_best[index]), "refined_best_iou": float(refined_best[index]), "final_best_iou": float(final_best[index]),
                    "raw_iou_01": bool(raw_best[index] > 0.1), "raw_iou_03": bool(raw_best[index] > 0.3), "raw_iou_05": bool(raw_best[index] > 0.5),
                    "refined_iou_05": bool(refined_best[index] > 0.5), "final_iou_05": bool(final_best[index] > 0.5),
                })
        del refined_masks

    metrics = pd.DataFrame(metric_rows); taxonomy = pd.DataFrame(taxonomy_rows).fillna(0)
    gt_frame = pd.DataFrame(gt_rows); candidates = pd.DataFrame(candidate_rows)
    metrics.to_csv(output_dir / "metrics_by_annotation.csv", index=False)
    taxonomy.to_csv(output_dir / "error_taxonomy_by_annotation.csv", index=False)
    gt_frame.to_csv(output_dir / "metrics_by_gt_instance.csv", index=False)
    candidates.to_csv(output_dir / "candidate_iou_transitions.csv", index=False)

    overall = _aggregate_stats(metrics)
    fold_rows = []
    for fold, frame in metrics.groupby("fold", sort=True): fold_rows.append({"fold": int(fold), **_aggregate_stats(frame)})
    fold_summary = pd.DataFrame(fold_rows); fold_summary.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    taxonomy_numeric = [column for column in taxonomy.columns if column not in {"file_name", "image_id", "fold"}]
    taxonomy_summary = {column: int(taxonomy[column].sum()) for column in taxonomy_numeric}
    write_json(output_dir / "error_taxonomy_summary.json", taxonomy_summary)

    size_summary = _stratified(gt_frame, "area", ["small", "medium", "large"], "quantile")
    radial_summary = _stratified(gt_frame, "radial_position", ["center", "middle", "limb"], [-np.inf, 0.45, 0.80, np.inf])
    elongation_summary = _stratified(gt_frame, "elongation", ["compact", "elongated", "very_elongated"], [-np.inf, 2.0, 4.0, np.inf])
    size_summary.to_csv(output_dir / "metrics_by_size.csv", index=False)
    radial_summary.to_csv(output_dir / "metrics_by_radial_position.csv", index=False)
    elongation_summary.to_csv(output_dir / "metrics_by_elongation.csv", index=False)

    labels = (quality_predictions["target_iou"].to_numpy() > 0.5).astype(int)
    probabilities = quality_predictions["quality_probability"].to_numpy()
    calibration = quality_predictions.assign(label=labels, score_bin=pd.cut(probabilities, np.linspace(0, 1, 11), include_lowest=True)).groupby("score_bin", observed=True).agg(
        candidates=("label", "size"), predicted_probability=("quality_probability", "mean"), observed_frequency=("label", "mean")
    ).reset_index()
    calibration["score_bin"] = calibration["score_bin"].astype(str)
    calibration.to_csv(output_dir / "quality_calibration.csv", index=False)
    calibration_metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "positive_rate": float(labels.mean()),
    }

    proposal = {
        "recall_at_0.1": float(gt_frame["raw_iou_01"].mean()),
        "recall_at_0.3": float(gt_frame["raw_iou_03"].mean()),
        "recall_at_0.5": float(gt_frame["raw_iou_05"].mean()),
        "refined_candidate_recall_at_0.5": float(gt_frame["refined_iou_05"].mean()),
        "final_recall_at_0.5": float(gt_frame["final_iou_05"].mean()),
    }
    transition_summary = {key: int(value) for key, value in transition_counts.items()}
    write_json(output_dir / "candidate_transition_summary.json", transition_summary)

    error_keys = ["missed_detection", "mask_iou_failure", "split_gt", "merge_prediction", "false_positive"]
    dominant_error = max(error_keys, key=lambda key: taxonomy_summary.get(key, 0))
    if proposal["recall_at_0.1"] < 0.90:
        bottleneck = "proposal_recall"
    elif proposal["refined_candidate_recall_at_0.5"] - proposal["recall_at_0.5"] < 0.02:
        bottleneck = "mask_refinement"
    elif overall["precision"] + 0.03 < overall["recall"]:
        bottleneck = "false_positive_selection"
    elif overall["recall"] + 0.03 < overall["precision"]:
        bottleneck = "false_negative_selection"
    else:
        bottleneck = "balanced_mask_quality_and_selection"
    summary = {
        "method": args.method, "quality_threshold": quality_threshold, "min_area": min_area,
        "official_oof": overall, "proposal_and_refinement": proposal,
        "candidate_transitions": transition_summary, "error_taxonomy": taxonomy_summary,
        "quality_calibration": calibration_metrics, "diagnosis": {"primary_bottleneck": bottleneck, "dominant_error": dominant_error},
    }
    write_json(output_dir / "diagnostic_summary.json", summary)

    plt = _load_plotting()
    _bar_plot(plt, fold_summary, "fold", "global_pq", "Official pooled PQ by held-out fold", charts_dir / "pq_by_fold.png", (0, max(0.55, fold_summary["global_pq"].max() * 1.15)))
    taxonomy_plot = pd.DataFrame({"error": error_keys, "count": [taxonomy_summary.get(key, 0) for key in error_keys]})
    _bar_plot(plt, taxonomy_plot, "error", "count", "M2 error taxonomy", charts_dir / "error_taxonomy.png")
    _bar_plot(plt, size_summary, "bin", "final_recall_05", "Final recall by GT size", charts_dir / "recall_by_size.png", (0, 1))
    _bar_plot(plt, radial_summary, "bin", "final_recall_05", "Final recall by radial position", charts_dir / "recall_by_radial.png", (0, 1))
    _bar_plot(plt, elongation_summary, "bin", "final_recall_05", "Final recall by elongation", charts_dir / "recall_by_elongation.png", (0, 1))
    transition_plot = pd.DataFrame({"transition": list(transition_summary), "count": list(transition_summary.values())})
    _bar_plot(plt, transition_plot, "transition", "count", "Raw-to-refined candidate IoU transitions", charts_dir / "candidate_transitions.png")
    figure, axis = plt.subplots(figsize=(5.5, 5.5)); axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.plot(calibration["predicted_probability"], calibration["observed_frequency"], marker="o", color="#b13f3f")
    axis.set(xlabel="Predicted probability", ylabel="Observed IoU>0.5 frequency", title="OOF quality calibration", xlim=(0, 1), ylim=(0, 1)); axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(charts_dir / "quality_calibration.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4.5)); axis.hist(metrics["pq"], bins=25, color="#3568a8", edgecolor="white")
    axis.set(title="PQ distribution across annotation records", xlabel="PQ", ylabel="Annotation records"); axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(charts_dir / "pq_distribution.png", dpi=160); plt.close(figure)

    physical = metrics.groupby("file_name").agg(mean_pq=("pq", "mean"), fold=("fold", "first")).reset_index()
    rescue_by_file = candidates[candidates["transition"] == "rescued_over_05"].groupby("file_name").size()
    case_groups = {
        "worst": physical.nsmallest(args.count, "mean_pq")["file_name"].tolist(),
        "best": physical.nlargest(args.count, "mean_pq")["file_name"].tolist(),
        "most_rescued": rescue_by_file.nlargest(args.count).index.tolist(),
    }
    medoid_metric_lookup = {(row.file_name, str(row.image_id)): float(row.pq) for row in metrics.itertuples(index=False)}
    for group_name, files in case_groups.items():
        tiles = []
        for file_name in files:
            image_id = medoid_by_file[file_name]; info = images[image_id]
            gt_rles = [polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]]
            raw_rles, raw_scores = raw_by_file[file_name]
            target = qualitative_dir / group_name / f"{Path(file_name).stem}.jpg"
            tiles.append(_render_case(image_dir / file_name, gt_rles, raw_rles, final_by_file[file_name], raw_scores, args.raw_conf, min_area, medoid_metric_lookup[(file_name, image_id)], target))
        _gallery(tiles, qualitative_dir / f"{group_name}_gallery.jpg")

    report = f"""# {args.method.upper()} OOF diagnostic report

## Official pooled metric

- Global PQ: {overall['global_pq']:.6f}
- SQ: {overall['sq']:.6f}
- RQ: {overall['rq']:.6f}
- Precision: {overall['precision']:.6f}
- Recall: {overall['recall']:.6f}
- TP / FP / FN: {overall['tp']} / {overall['fp']} / {overall['fn']}

## Proposal and refinement ceiling

- Raw proposal recall IoU>0.1: {proposal['recall_at_0.1']:.6f}
- Raw proposal recall IoU>0.3: {proposal['recall_at_0.3']:.6f}
- Raw proposal recall IoU>0.5: {proposal['recall_at_0.5']:.6f}
- Refined candidate recall IoU>0.5: {proposal['refined_candidate_recall_at_0.5']:.6f}
- Final selected recall IoU>0.5: {proposal['final_recall_at_0.5']:.6f}

## Automated diagnosis

- Primary bottleneck: **{bottleneck}**
- Dominant observed error: **{dominant_error}**
- Quality ROC-AUC / AP / Brier: {calibration_metrics['roc_auc']:.4f} / {calibration_metrics['average_precision']:.4f} / {calibration_metrics['brier_score']:.4f}

See the CSV tables, charts, and qualitative galleries in this directory before choosing the next method.
"""
    (output_dir / "diagnostic_report.md").write_text(report)
    print(json.dumps(summary, indent=2))
    print(f"Diagnostic report: {output_dir / 'diagnostic_report.md'}")
    print(f"Visual galleries: {qualitative_dir}")


if __name__ == "__main__":
    main()
