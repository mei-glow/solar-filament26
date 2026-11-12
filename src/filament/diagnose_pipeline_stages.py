from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from skimage.morphology import skeletonize
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import write_json
from .oof_bank import OOFProposalBank
from .p5_selector import decode_cluster_first
from .postprocess import polygon_to_rle, rle_from_json


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _matching(iou: np.ndarray) -> list[tuple[int, int, float]]:
    predictions, targets = iou.shape
    if not predictions or not targets:
        return []
    valid = iou > 0.5
    bonus = min(predictions, targets) + 1.0
    rows, columns = linear_sum_assignment(-(valid.astype(np.float64) * bonus + iou))
    return [(int(row), int(column), float(iou[row, column])) for row, column in zip(rows, columns) if valid[row, column]]


def _stats(iou: np.ndarray) -> dict[str, float | int]:
    pairs = _matching(iou)
    tp = len(pairs); fp = iou.shape[0] - tp; fn = iou.shape[1] - tp
    iou_sum = float(sum(value for _, _, value in pairs))
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "iou_sum": iou_sum, "tp": tp, "fp": fp, "fn": fn,
        "pq": _safe_div(iou_sum, denominator), "sq": _safe_div(iou_sum, tp),
        "rq": _safe_div(tp, denominator),
    }


class _UnionFind:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _load_probability(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path) as payload:
        probability = payload["probability"].astype(np.float32)
    return probability[0] if probability.ndim == 3 else probability


def _aggregate(rows: pd.DataFrame) -> dict[str, float]:
    iou_sum = float(rows["iou_sum"].sum()); tp = int(rows["tp"].sum())
    fp = int(rows["fp"].sum()); fn = int(rows["fn"].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "global_pq": _safe_div(iou_sum, denominator), "mean_pq": float(rows["pq"].mean()),
        "sq": _safe_div(iou_sum, tp), "rq": _safe_div(tp, denominator),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring pandas' optional tabulate dependency."""
    if frame.empty:
        return "_No data available._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = [str(column) for column in display.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in display.fillna("").to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _plot_detector(detector_gt: pd.DataFrame, path: Path) -> None:
    thresholds = [0.1, 0.3, 0.5]
    overall = [(detector_gt["best_raw_iou"] > value).mean() for value in thresholds]
    size = detector_gt.groupby("size_bin", observed=True)["best_raw_iou"].apply(
        lambda values: pd.Series({str(t): float((values > t).mean()) for t in thresholds})
    ).unstack()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar([str(value) for value in thresholds], overall)
    axes[0].set(title="Detector proposal recall", xlabel="IoU threshold", ylabel="Recall", ylim=(0, 1))
    x = np.arange(len(size)); width = 0.24
    for index, threshold in enumerate(thresholds):
        axes[1].bar(x + (index - 1) * width, size[str(threshold)], width, label=f"IoU>{threshold}")
    axes[1].set_xticks(x, size.index); axes[1].set_ylim(0, 1)
    axes[1].set(title="Recall by GT area", ylabel="Recall"); axes[1].legend()
    _save_figure(path)


def _plot_semantic(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.groupby("fold")[["dice", "iou", "skeleton_recall"]].mean()
    summary.plot(kind="bar", figsize=(9, 4), ylim=(0, 1), title="Semantic OOF metrics by fold")
    plt.ylabel("Mean metric"); plt.xticks(rotation=0); _save_figure(path)


def _plot_sira(frame: pd.DataFrame, path: Path) -> None:
    values = frame["delta_same_anchor_iou_mean"].clip(-0.5, 0.5)
    plt.figure(figsize=(8, 4)); plt.hist(values, bins=60)
    plt.axvline(0, color="black", linewidth=1); plt.xlabel("Refined IoU − raw IoU")
    plt.ylabel("Candidates"); plt.title("SIRA paired-IoU change")
    _save_figure(path)


def _plot_pair(frame: pd.DataFrame, path: Path) -> None:
    target = (frame["same_gt_frequency_03"].to_numpy() > 0).astype(int)
    score = frame["predicted_duplicate_probability"].to_numpy()
    precision, recall, _ = precision_recall_curve(target, score)
    bins = pd.cut(score, np.linspace(0, 1, 11), include_lowest=True)
    calibration = pd.DataFrame({"bin": bins, "target": target, "score": score}).groupby("bin", observed=True).agg(
        predicted=("score", "mean"), observed=("target", "mean"), count=("target", "size")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(recall, precision); axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Pair-head PR curve")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].plot(calibration["predicted"], calibration["observed"], marker="o")
    axes[1].set(xlabel="Predicted duplicate probability", ylabel="Observed same-GT rate", xlim=(0, 1), ylim=(0, 1), title="Pair calibration")
    _save_figure(path)


def _plot_quality(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    true_quality = frame[["raw_anchor_iou_mean", "refined_same_anchor_iou_mean"]].max(axis=1)
    predicted = frame[["predicted_raw_paired_iou", "predicted_refined_paired_iou"]].max(axis=1)
    bins = pd.cut(predicted, np.linspace(0, 1, 11), include_lowest=True)
    calibration = pd.DataFrame({"bin": bins, "true": true_quality, "predicted": predicted}).groupby("bin", observed=True).agg(
        predicted=("predicted", "mean"), true_iou=("true", "mean"), match_rate=("true", lambda values: float((values > 0.5).mean())), count=("true", "size")
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(predicted, true_quality, s=3, alpha=0.12)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="Predicted paired IoU", ylabel="True paired IoU", xlim=(0, 1), ylim=(0, 1), title="Candidate-quality regression")
    axes[1].plot(calibration["predicted"], calibration["match_rate"], marker="o")
    axes[1].axvline(0.45, linestyle="--", color="gray")
    axes[1].set(xlabel="Predicted paired IoU", ylabel="Observed P(IoU>0.5)", xlim=(0, 1), ylim=(0, 1), title="Quality calibration")
    _save_figure(path); return calibration


def _plot_decoder(frame: pd.DataFrame, path: Path) -> None:
    values = frame.sort_values("global_pq")
    plt.figure(figsize=(9, 4)); plt.barh(values["stage"], values["global_pq"])
    plt.xlabel("Official pooled PQ"); plt.xlim(0, max(0.6, values["global_pq"].max() * 1.08))
    plt.title("End-to-end stage ablation"); _save_figure(path)


def _contours(axis, rles: list[dict], color: str, linewidth: float = 0.7, alpha: float = 0.8) -> None:
    for rle in rles:
        mask = mask_utils.decode(rle).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = contour[:, 0]
            axis.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def _render_case(
    target: Path,
    image_path: Path,
    gt: list[dict],
    raw_instances: list[dict],
    refined_instances: list[dict],
    semantic: np.ndarray | None,
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    selected: list[dict],
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read diagnostic image: {image_path}")
    raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
    refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
    selected_rles = [
        (raw_rles if item["action"] == "raw" else refined_rles)[int(item["candidate_index"])] for item in selected
    ]
    iou = mask_utils.iou(selected_rles, gt, [0] * len(gt)) if selected_rles and gt else np.zeros((len(selected_rles), len(gt)))
    matches = _matching(iou); matched_pred = {left for left, _, _ in matches}; matched_gt = {right for _, right, _ in matches}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis in axes.flat:
        axis.imshow(image, cmap="gray"); axis.axis("off")
    axes[0, 0].set_title("Input")
    _contours(axes[0, 1], gt, "lime", 1.2); axes[0, 1].set_title(f"GT instances ({len(gt)})")
    if semantic is not None:
        axes[0, 2].imshow(semantic, cmap="magma", alpha=0.65, vmin=0, vmax=1)
    axes[0, 2].set_title("Semantic probability")
    _contours(axes[1, 0], raw_rles, "yellow", 0.5, 0.45)
    _contours(axes[1, 0], refined_rles, "magenta", 0.5, 0.45)
    axes[1, 0].set_title(f"Raw yellow / refined magenta ({len(raw_rles)})")
    boxes = np.asarray(mask_utils.toBbox(raw_rles), np.float32).reshape(-1, 4) if raw_rles else np.empty((0, 4))
    centers = boxes[:, :2] + boxes[:, 2:] / 2 if len(boxes) else np.empty((0, 2))
    for row in pairs.itertuples(index=False):
        if float(row.predicted_duplicate_probability) >= 0.5:
            left, right = int(row.left_candidate_index), int(row.right_candidate_index)
            axes[1, 1].plot([centers[left, 0], centers[right, 0]], [centers[left, 1], centers[right, 1]], color="cyan", linewidth=0.5, alpha=0.35)
    selected_indices = {int(item["candidate_index"]) for item in selected}
    if len(centers):
        colors = ["lime" if index in selected_indices else "orange" for index in range(len(centers))]
        axes[1, 1].scatter(centers[:, 0], centers[:, 1], c=colors, s=12)
    axes[1, 1].set_title("Duplicate graph; selected nodes green")
    _contours(axes[1, 2], [selected_rles[index] for index in matched_pred], "lime", 1.2)
    _contours(axes[1, 2], [rle for index, rle in enumerate(selected_rles) if index not in matched_pred], "red", 1.2)
    _contours(axes[1, 2], [rle for index, rle in enumerate(gt) if index not in matched_gt], "cyan", 1.2)
    axes[1, 2].set_title(f"Final: TP {len(matches)} / FP {len(selected_rles)-len(matches)} / FN {len(gt)-len(matches)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(target)


def _representative_ranking(candidates: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for file_name, current in candidates.groupby("file_name", sort=False):
        current = current.sort_values("candidate_index")
        indices = current["candidate_index"].astype(int).tolist()
        union = _UnionFind(indices)
        local_pairs = pairs[pairs["file_name"] == file_name]
        for row in local_pairs.itertuples(index=False):
            if float(row.same_gt_frequency_03) > 0:
                union.union(int(row.left_candidate_index), int(row.right_candidate_index))
        groups: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            groups[union.find(index)].append(index)
        indexed = current.set_index("candidate_index")
        for members in groups.values():
            if len(members) < 2:
                continue
            subset = indexed.loc[members]
            true_quality = subset[["raw_anchor_iou_mean", "refined_same_anchor_iou_mean"]].max(axis=1)
            predicted = subset[["predicted_raw_paired_iou", "predicted_refined_paired_iou"]].max(axis=1)
            chosen = int(predicted.idxmax()); oracle = int(true_quality.idxmax())
            rows.append({
                "file_name": file_name, "component_size": len(members), "chosen_candidate": chosen,
                "oracle_candidate": oracle, "top1_correct": chosen == oracle,
                "true_best_iou": float(true_quality.max()), "chosen_true_iou": float(true_quality.loc[chosen]),
                "ranking_regret": float(true_quality.max() - true_quality.loc[chosen]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified stage-by-stage OOF audit for the M2/P5.3 pipeline")
    add_path_args(parser)
    parser.add_argument("--run-name", default="pipeline_stage_audit_m2")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--p5-1-run", default="p5_1_oracle_supervision_m2")
    parser.add_argument("--p5-2-run", default="p5_2_graph_selector_m2")
    parser.add_argument("--p5-3-run", default="p5_3_cluster_first_m2")
    parser.add_argument("--raw-bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--sira-run-name", default="m2_sira_oof")
    parser.add_argument("--semantic-cache-name", default="m1_oof_s2")
    parser.add_argument("--gallery-size", type=int, default=12)
    parser.add_argument("--skip-gallery", action="store_true")
    args = parser.parse_args()

    paths = paths_from_args(args); annotation_path, image_dir, _ = discover_data(paths)
    manifest_path = (args.manifest or paths.results_root / "fold_manifest_s3020.csv").resolve()
    manifest = pd.read_csv(manifest_path)
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]:
        annotations[str(item["image_id"])].append(item)

    p51 = paths.results_root / args.p5_1_run; p52 = paths.results_root / args.p5_2_run; p53 = paths.results_root / args.p5_3_run
    candidates = pd.read_csv(p53 / "oof_cluster_candidates.csv")
    pairs = pd.read_csv(p52 / "oof_pair_predictions.csv.gz")
    supervision = pd.read_csv(p51 / "candidate_supervision.csv")
    tuning2 = pd.read_csv(p52 / "p5_2_tuning.csv")
    tuning3 = pd.read_csv(p53 / "p5_3_tuning.csv")
    best3 = json.loads((p53 / "best_p5_cluster_selector.json").read_text())
    output = paths.experiment_results(args.run_name); charts = output / "charts"; galleries = output / "galleries"
    charts.mkdir(parents=True, exist_ok=True)

    candidate_by_file = {name: group.sort_values("candidate_index") for name, group in candidates.groupby("file_name", sort=False)}
    pair_by_file = {name: group for name, group in pairs.groupby("file_name", sort=False)}
    detector_rows, semantic_rows, final_rows, visual_payloads = [], [], [], {}
    grouped = list(manifest.groupby("file_name", sort=True))
    for file_name, records in tqdm(grouped, desc="Unified pipeline audit", unit="image"):
        fold = int(records["fold"].iloc[0])
        raw_payload = OOFProposalBank(paths.artifact_root / "predictions" / "oof" / args.raw_bank_name / f"fold{fold}").load(file_name, decode_masks=False)
        refined_payload = OOFProposalBank(paths.artifact_root / "predictions" / "sira_oof" / args.sira_run_name / f"fold{fold}").load(file_name, decode_masks=False)
        raw_instances, refined_instances = raw_payload["instances"], refined_payload["instances"]
        raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
        refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
        current_candidates = candidate_by_file.get(file_name, candidates.iloc[0:0])
        current_pairs = pair_by_file.get(file_name, pairs.iloc[0:0])
        selected = decode_cluster_first(
            current_candidates, current_pairs, float(best3["quality_threshold"]),
            float(best3["duplicate_threshold"]), int(best3["min_area"]),
        )
        selected_rles = [
            (raw_rles if item["action"] == "raw" else refined_rles)[int(item["candidate_index"])] for item in selected
        ]
        semantic_path = paths.artifact_root / "predictions" / "semantic_oof" / args.semantic_cache_name / f"fold{fold}" / f"{Path(file_name).stem}.npz"
        semantic = _load_probability(semantic_path)
        image_stats = []
        first_gt, first_info = [], None
        for record in records.itertuples(index=False):
            image_id = str(record.image_id); info = images[image_id]
            gt = [polygon_to_rle(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]]
            if first_info is None:
                first_gt, first_info = gt, info
            raw_iou = mask_utils.iou(raw_rles, gt, [0] * len(gt)) if raw_rles and gt else np.zeros((len(raw_rles), len(gt)))
            best = raw_iou.max(axis=0) if len(raw_rles) else np.zeros(len(gt))
            for gt_index, (rle, best_iou) in enumerate(zip(gt, best)):
                bbox = np.asarray(mask_utils.toBbox(rle), np.float32)
                center_x, center_y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                radial = np.hypot(center_x - info["width"] / 2, center_y - info["height"] / 2) / max(min(info["height"], info["width"]) / 2, 1)
                detector_rows.append({
                    "file_name": file_name, "fold": fold, "image_id": image_id, "gt_index": gt_index,
                    "gt_area": float(mask_utils.area(rle)), "radial_position": float(radial), "best_raw_iou": float(best_iou),
                    "candidates": len(raw_rles),
                })
            final_iou = mask_utils.iou(selected_rles, gt, [0] * len(gt)) if selected_rles and gt else np.zeros((len(selected_rles), len(gt)))
            values = _stats(final_iou); image_stats.append(values)
            final_rows.append({"file_name": file_name, "fold": fold, "image_id": image_id, "predictions": len(selected_rles), **values})
            if semantic is not None:
                gt_union = np.zeros((info["height"], info["width"]), bool)
                for rle in gt:
                    gt_union |= mask_utils.decode(rle).astype(bool)
                prediction = semantic >= 0.5
                intersection = float((prediction & gt_union).sum()); union = float((prediction | gt_union).sum())
                target_skeleton = skeletonize(gt_union)
                semantic_rows.append({
                    "file_name": file_name, "fold": fold, "image_id": image_id,
                    "dice": _safe_div(2 * intersection, prediction.sum() + gt_union.sum()),
                    "iou": _safe_div(intersection, union),
                    "skeleton_recall": _safe_div((prediction & target_skeleton).sum(), target_skeleton.sum()),
                })
        aggregate = {
            key: sum(float(item[key]) for item in image_stats) for key in ("iou_sum", "tp", "fp", "fn")
        }
        denominator = aggregate["tp"] + 0.5 * aggregate["fp"] + 0.5 * aggregate["fn"]
        visual_payloads[file_name] = {
            "fold": fold, "gt": first_gt, "semantic": semantic, "raw": raw_instances, "refined": refined_instances,
            "candidates": current_candidates, "pairs": current_pairs, "selected": selected,
            "pq": _safe_div(aggregate["iou_sum"], denominator), "fp": aggregate["fp"], "fn": aggregate["fn"],
            "image_path": image_dir / file_name,
        }

    detector_gt = pd.DataFrame(detector_rows)
    area_cut = detector_gt["gt_area"].quantile([1 / 3, 2 / 3]).to_numpy()
    detector_gt["size_bin"] = pd.cut(detector_gt["gt_area"], [-np.inf, area_cut[0], area_cut[1], np.inf], labels=["small", "medium", "large"])
    semantic_frame = pd.DataFrame(semantic_rows); final_frame = pd.DataFrame(final_rows)
    detector_gt.to_csv(output / "detector_gt_audit.csv", index=False)
    semantic_frame.to_csv(output / "semantic_oof_audit.csv", index=False)
    final_frame.to_csv(output / "final_by_annotation.csv", index=False)
    supervision.to_csv(output / "sira_candidate_audit.csv", index=False)

    pair_target = (pairs["same_gt_frequency_03"].to_numpy() > 0).astype(int)
    pair_score = pairs["predicted_duplicate_probability"].to_numpy()
    pair_pred = pair_score >= 0.5
    tp = int((pair_target & pair_pred).sum()); fp = int(((1 - pair_target) & pair_pred).sum())
    fn = int((pair_target & ~pair_pred).sum())
    pair_summary = {
        "roc_auc": float(roc_auc_score(pair_target, pair_score)), "pr_auc": float(average_precision_score(pair_target, pair_score)),
        "precision_at_05": _safe_div(tp, tp + fp), "recall_at_05": _safe_div(tp, tp + fn), "tp": tp, "fp": fp, "fn": fn,
    }
    true_quality = candidates[["raw_anchor_iou_mean", "refined_same_anchor_iou_mean"]].max(axis=1).to_numpy()
    predicted_quality = candidates[["predicted_raw_paired_iou", "predicted_refined_paired_iou"]].max(axis=1).to_numpy()
    match_target = (true_quality > 0.5).astype(int)
    quality_summary = {
        "roc_auc": float(roc_auc_score(match_target, predicted_quality)), "pr_auc": float(average_precision_score(match_target, predicted_quality)),
        "mae": float(np.abs(true_quality - predicted_quality).mean()),
    }
    ranking = _representative_ranking(candidates, pairs)
    ranking.to_csv(output / "cluster_representative_audit.csv", index=False)
    quality_summary.update({
        "cluster_top1_accuracy": float(ranking["top1_correct"].mean()) if len(ranking) else float("nan"),
        "mean_ranking_regret": float(ranking["ranking_regret"].mean()) if len(ranking) else float("nan"),
        "components_evaluated": int(len(ranking)),
    })

    best2 = tuning2.iloc[0]
    best3_row = tuning3.iloc[0]
    no_graph_rows2 = tuning2[tuning2["duplicate_threshold"] > 1].sort_values("global_pq", ascending=False)
    no_graph_rows3 = tuning3[tuning3["duplicate_threshold"] > 1].sort_values("global_pq", ascending=False)
    no_graph2 = no_graph_rows2.iloc[0] if len(no_graph_rows2) else best2
    no_graph3 = no_graph_rows3.iloc[0] if len(no_graph_rows3) else best3_row
    decoder_rows = [
        {"stage": "P5.2 candidate-only", **{name: no_graph2[name] for name in ("global_pq", "sq", "rq", "tp", "fp", "fn")}},
        {"stage": "P5.2 + duplicate graph", **{name: best2[name] for name in ("global_pq", "sq", "rq", "tp", "fp", "fn")}},
        {"stage": "P5.3 paired-IoU only", **{name: no_graph3[name] for name in ("global_pq", "sq", "rq", "tp", "fp", "fn")}},
        {"stage": "P5.3 cluster-first", **{name: best3_row[name] for name in ("global_pq", "sq", "rq", "tp", "fp", "fn")}},
    ]
    oracle_path = paths.results_root / "oracle_o1_o5_m2" / "oracle_decomposition.json"
    if oracle_path.is_file():
        oracle = json.loads(oracle_path.read_text())
        for name in ("O1_current_m4", "O4_raw_refined_oracle_set"):
            if name in oracle:
                values = oracle[name]
                decoder_rows.append({"stage": name, **{key: values[key] for key in ("global_pq", "sq", "rq", "tp", "fp", "fn")}})
    decoder = pd.DataFrame(decoder_rows); decoder.to_csv(output / "decoder_stage_ablation.csv", index=False)

    _plot_detector(detector_gt, charts / "detector_recall.png")
    if len(semantic_frame): _plot_semantic(semantic_frame, charts / "semantic_by_fold.png")
    _plot_sira(supervision, charts / "sira_delta_iou.png")
    _plot_pair(pairs, charts / "pair_head.png")
    quality_calibration = _plot_quality(candidates, charts / "quality_head.png")
    quality_calibration.to_csv(output / "quality_calibration.csv", index=False)
    _plot_decoder(decoder, charts / "decoder_ablation.png")

    damage_by_file = supervision.groupby("file_name")["damage_crossing_frequency"].sum()
    false_edges = pairs.assign(false_edge=(pair_pred & (pair_target == 0))).groupby("file_name")["false_edge"].sum()
    selections = {
        "worst_pq": sorted(visual_payloads, key=lambda name: visual_payloads[name]["pq"])[:args.gallery_size],
        "highest_fp": sorted(visual_payloads, key=lambda name: visual_payloads[name]["fp"], reverse=True)[:args.gallery_size],
        "highest_fn": sorted(visual_payloads, key=lambda name: visual_payloads[name]["fn"], reverse=True)[:args.gallery_size],
        "sira_damage": damage_by_file.sort_values(ascending=False).head(args.gallery_size).index.tolist(),
        "false_duplicate_edges": false_edges.sort_values(ascending=False).head(args.gallery_size).index.tolist(),
    }
    if not args.skip_gallery:
        rendered: dict[str, Path] = {}
        for category, names in selections.items():
            for rank_index, name in enumerate(names, start=1):
                source = rendered.get(name)
                target = galleries / category / f"{rank_index:02d}_{Path(name).stem}.png"
                if source is not None:
                    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target); continue
                payload = visual_payloads[name]
                _render_case(
                    target, payload["image_path"], payload["gt"], payload["raw"], payload["refined"], payload["semantic"],
                    payload["candidates"], payload["pairs"], payload["selected"],
                )
                rendered[name] = target

    detector_summary = {
        f"recall_at_{threshold}": float((detector_gt["best_raw_iou"] > threshold).mean()) for threshold in (0.1, 0.3, 0.5)
    }
    semantic_summary = {
        key: float(semantic_frame[key].mean()) for key in ("dice", "iou", "skeleton_recall")
    } if len(semantic_frame) else {"available": False}
    sira_summary = {
        "mean_delta_iou": float(supervision["delta_same_anchor_iou_mean"].mean()),
        "gain_crossing_rate": float((supervision["gain_crossing_frequency"] > 0).mean()),
        "damage_crossing_rate": float((supervision["damage_crossing_frequency"] > 0).mean()),
        "identity_switch_rate": float((supervision["identity_switch_frequency"] > 0).mean()),
    }
    final_summary = _aggregate(final_frame)
    module_rows = [
        {"module": "detector", "status": "KEEP", "primary_metric": "proposal_recall_at_01", "value": detector_summary["recall_at_0.1"]},
        {"module": "semantic", "status": "UNKNOWN", "primary_metric": "skeleton_recall", "value": semantic_summary.get("skeleton_recall", np.nan)},
        {"module": "sira_refiner", "status": "FIX_OR_GATE", "primary_metric": "mean_paired_delta_iou", "value": sira_summary["mean_delta_iou"]},
        {"module": "pair_head", "status": "KEEP", "primary_metric": "pair_pr_auc", "value": pair_summary["pr_auc"]},
        {"module": "quality_head", "status": "FIX", "primary_metric": "cluster_top1_accuracy", "value": quality_summary["cluster_top1_accuracy"]},
        {"module": "cluster_decoder", "status": "FIX_OR_SIMPLIFY", "primary_metric": "global_pq", "value": final_summary["global_pq"]},
    ]
    pd.DataFrame(module_rows).to_csv(output / "module_scorecard.csv", index=False)
    summary = {
        "detector": detector_summary, "semantic": semantic_summary, "sira": sira_summary,
        "pair_head": pair_summary, "quality_head": quality_summary, "final": final_summary,
        "configuration": {"p5_3_best": best3, "manifest": str(manifest_path), "gallery_size": args.gallery_size},
    }
    write_json(output / "pipeline_audit_summary.json", summary)

    semantic_chart = "![Semantic metrics](charts/semantic_by_fold.png)" if len(semantic_frame) else "_Semantic OOF cache was not available._"
    report = f"""# Pipeline stage audit

## Module scorecard

{_markdown_table(pd.DataFrame(module_rows))}

## Detector

{_markdown_table(pd.DataFrame([detector_summary]))}

![Detector recall](charts/detector_recall.png)

## Semantic

{_markdown_table(pd.DataFrame([semantic_summary]))}

{semantic_chart}

## SIRA refiner

{_markdown_table(pd.DataFrame([sira_summary]))}

![SIRA paired delta](charts/sira_delta_iou.png)

## Pair head

{_markdown_table(pd.DataFrame([pair_summary]))}

![Pair diagnostics](charts/pair_head.png)

## Candidate quality and representative ranking

{_markdown_table(pd.DataFrame([quality_summary]))}

![Quality diagnostics](charts/quality_head.png)

## Decoder and end-to-end ablation

{_markdown_table(decoder)}

![Decoder ablation](charts/decoder_ablation.png)

## Qualitative galleries

- `galleries/worst_pq/`
- `galleries/highest_fp/`
- `galleries/highest_fn/`
- `galleries/sira_damage/`
- `galleries/false_duplicate_edges/`
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(module_rows).to_string(index=False), flush=True)
    print("\nFinal:", json.dumps(final_summary, indent=2), flush=True)
    print(f"Report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
