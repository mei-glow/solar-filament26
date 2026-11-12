from __future__ import annotations

import numpy as np
import pandas as pd


ACTIONS = ("reject", "raw", "refined")
PAIR_FEATURES = [
    "raw_mask_iou",
    "refined_mask_iou",
    "max_cross_variant_iou",
    "raw_box_iou",
    "normalized_center_distance",
    "absolute_log_area_ratio",
    "detector_score_difference",
    "semantic_support_difference",
]
CONTEXT_FEATURES = [
    "candidate_count_log1p",
    "detector_score_rank",
    "semantic_support_rank",
    "raw_refined_iou_rank",
    "raw_area_rank",
    "refined_area_rank",
    "overlap_neighbor_count_005",
    "overlap_neighbor_count_030",
    "overlap_neighbor_count_050",
    "maximum_neighbor_overlap",
    "mean_neighbor_overlap_005",
    "detector_margin_to_strong_neighbor",
]


def pair_prefilter(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return (
        frame["max_cross_variant_iou"].to_numpy(np.float32) >= threshold
    ) | (
        frame["raw_box_iou"].to_numpy(np.float32) >= threshold
    )


def build_pair_features(
    file_name: str,
    raw_instances: list[dict],
    refined_instances: list[dict],
    candidate_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build the same inference-safe pair features used by P5.1."""
    from pycocotools import mask as mask_utils

    from .postprocess import rle_from_json

    count = len(raw_instances)
    columns = ["file_name", "left_candidate_index", "right_candidate_index"] + PAIR_FEATURES
    if count < 2:
        return pd.DataFrame(columns=columns)
    raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
    refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
    raw_mask_iou = mask_utils.iou(raw_rles, raw_rles, [0] * count)
    refined_mask_iou = mask_utils.iou(refined_rles, refined_rles, [0] * count)
    cross = mask_utils.iou(raw_rles, refined_rles, [0] * count)
    raw_boxes = np.asarray(mask_utils.toBbox(raw_rles), np.float32).reshape(-1, 4)
    x, y, width, height = raw_boxes.T
    x2, y2 = x + width, y + height
    inter_width = np.maximum(0.0, np.minimum(x2[:, None], x2[None]) - np.maximum(x[:, None], x[None]))
    inter_height = np.maximum(0.0, np.minimum(y2[:, None], y2[None]) - np.maximum(y[:, None], y[None]))
    intersection = inter_width * inter_height
    area = width * height
    box_iou = intersection / np.maximum(area[:, None] + area[None] - intersection, 1e-6)
    centers = raw_boxes[:, :2] + raw_boxes[:, 2:] / 2.0
    height_px, width_px = map(float, raw_rles[0]["size"])
    diagonal = max(float(np.hypot(height_px, width_px)), 1.0)
    center_distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=2) / diagonal
    raw_area = np.asarray(mask_utils.area(raw_rles), np.float32).reshape(-1)
    detector = candidate_features["detector_score"].to_numpy(np.float32)
    semantic = candidate_features["semantic_support"].to_numpy(np.float32)
    rows = []
    for left in range(count):
        for right in range(left + 1, count):
            rows.append({
                "file_name": file_name,
                "left_candidate_index": left,
                "right_candidate_index": right,
                "raw_mask_iou": float(raw_mask_iou[left, right]),
                "refined_mask_iou": float(refined_mask_iou[left, right]),
                "max_cross_variant_iou": float(max(
                    cross[left, right], cross[right, left], raw_mask_iou[left, right], refined_mask_iou[left, right]
                )),
                "raw_box_iou": float(box_iou[left, right]),
                "normalized_center_distance": float(center_distance[left, right]),
                "absolute_log_area_ratio": float(abs(np.log((raw_area[left] + 1.0) / (raw_area[right] + 1.0)))),
                "detector_score_difference": float(abs(detector[left] - detector[right])),
                "semantic_support_difference": float(abs(semantic[left] - semantic[right])),
            })
    return pd.DataFrame(rows, columns=columns)


def add_candidate_context(candidates: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Add inference-safe within-image ranking and overlap-neighborhood features."""
    output = candidates.copy()
    grouped = output.groupby("file_name", sort=False)
    output["candidate_count_log1p"] = np.log1p(grouped["candidate_index"].transform("size"))
    for source, target in (
        ("detector_score", "detector_score_rank"),
        ("semantic_support", "semantic_support_rank"),
        ("raw_refined_iou", "raw_refined_iou_rank"),
        ("raw_area", "raw_area_rank"),
        ("refined_area", "refined_area_rank"),
    ):
        output[target] = grouped[source].rank(method="average", pct=True)

    size = len(output)
    count_005 = np.zeros(size, np.float32)
    count_030 = np.zeros(size, np.float32)
    count_050 = np.zeros(size, np.float32)
    overlap_sum = np.zeros(size, np.float32)
    maximum = np.zeros(size, np.float32)
    strong_neighbor_score = np.full(size, -np.inf, np.float32)
    key_to_row = {
        (str(row.file_name), int(row.candidate_index)): position
        for position, row in enumerate(output.itertuples(index=False))
    }
    detector_scores = output["detector_score"].to_numpy(np.float32)

    for row in pairs.itertuples(index=False):
        left = key_to_row[(str(row.file_name), int(row.left_candidate_index))]
        right = key_to_row[(str(row.file_name), int(row.right_candidate_index))]
        overlap = float(max(row.max_cross_variant_iou, row.raw_box_iou))
        if overlap < 0.05:
            continue
        for index in (left, right):
            count_005[index] += 1
            overlap_sum[index] += overlap
            maximum[index] = max(maximum[index], overlap)
        if overlap >= 0.30:
            count_030[left] += 1; count_030[right] += 1
            strong_neighbor_score[left] = max(strong_neighbor_score[left], detector_scores[right])
            strong_neighbor_score[right] = max(strong_neighbor_score[right], detector_scores[left])
        if overlap >= 0.50:
            count_050[left] += 1; count_050[right] += 1

    no_strong_neighbor = ~np.isfinite(strong_neighbor_score)
    strong_neighbor_score[no_strong_neighbor] = detector_scores[no_strong_neighbor]
    output["overlap_neighbor_count_005"] = count_005
    output["overlap_neighbor_count_030"] = count_030
    output["overlap_neighbor_count_050"] = count_050
    output["maximum_neighbor_overlap"] = maximum
    output["mean_neighbor_overlap_005"] = overlap_sum / np.maximum(count_005, 1.0)
    output["detector_margin_to_strong_neighbor"] = detector_scores - strong_neighbor_score
    return output


def normalize_action_predictions(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, np.float64), 1e-6, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


class _UnionFind:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def decode_candidate_set(
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    selection_threshold: float,
    duplicate_threshold: float,
    min_area: int,
    mode: str,
) -> list[dict[str, float | int | str]]:
    """Choose one raw/refined action per proposal and suppress predicted duplicates."""
    if mode not in {"edge", "component"}:
        raise ValueError(f"Unknown decoder mode: {mode}")
    items = []
    for row in candidates.itertuples(index=False):
        available = []
        if float(row.raw_area) >= min_area:
            available.append(("raw", float(row.predicted_p_raw)))
        if float(row.refined_area) >= min_area:
            available.append(("refined", float(row.predicted_p_refined)))
        if not available:
            continue
        available_mass = float(row.predicted_p_reject) + sum(value for _, value in available)
        selection_score = sum(value for _, value in available) / max(available_mass, 1e-8)
        if selection_score < selection_threshold:
            continue
        action, action_score = max(available, key=lambda value: value[1])
        items.append({
            "candidate_index": int(row.candidate_index),
            "action": action,
            "selection_score": selection_score,
            "action_score": action_score,
            "detector_score": float(row.detector_score),
        })
    items.sort(
        key=lambda item: (item["selection_score"], item["action_score"], item["detector_score"]),
        reverse=True,
    )
    if not items:
        return []

    eligible = {int(item["candidate_index"]) for item in items}
    conflicts: dict[int, set[int]] = {index: set() for index in eligible}
    for row in pairs.itertuples(index=False):
        left, right = int(row.left_candidate_index), int(row.right_candidate_index)
        if left in eligible and right in eligible and float(row.predicted_duplicate_probability) >= duplicate_threshold:
            conflicts[left].add(right); conflicts[right].add(left)

    if mode == "edge":
        selected = []
        selected_indices: set[int] = set()
        for item in items:
            index = int(item["candidate_index"])
            if conflicts[index].isdisjoint(selected_indices):
                selected.append(item); selected_indices.add(index)
        return selected

    union = _UnionFind(sorted(eligible))
    for left, neighbors in conflicts.items():
        for right in neighbors:
            union.union(left, right)
    seen_roots: set[int] = set()
    selected = []
    for item in items:
        root = union.find(int(item["candidate_index"]))
        if root not in seen_roots:
            selected.append(item); seen_roots.add(root)
    return selected
