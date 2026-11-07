from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .postprocess import mask_to_rle_object, resolve_instances_scored, rle_from_json


BASE_FEATURES = [
    "detector_score", "mean_probability", "q90_probability", "seed_agreement",
    "semantic_support", "boundary_uncertainty", "elongation", "radial_position",
    "area", "components",
]

FEATURES = BASE_FEATURES + [
    "raw_area", "refined_area", "log_area_ratio", "raw_refined_iou",
    "raw_box_fill", "refined_box_fill", "raw_refined_box_iou",
]


def _box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lw, lh = left.T; rx, ry, rw, rh = right.T
    lx2, ly2, rx2, ry2 = lx + lw, ly + lh, rx + rw, ry + rh
    inter_w = np.maximum(0.0, np.minimum(lx2, rx2) - np.maximum(lx, rx))
    inter_h = np.maximum(0.0, np.minimum(ly2, ry2) - np.maximum(ly, ry))
    intersection = inter_w * inter_h
    union = lw * lh + rw * rh - intersection
    return intersection / np.maximum(union, 1e-6)


def build_candidate_features(raw_instances: list[dict], refined_instances: list[dict]) -> pd.DataFrame:
    if len(raw_instances) != len(refined_instances):
        raise RuntimeError(f"Raw/refined candidate count mismatch: {len(raw_instances)} != {len(refined_instances)}")
    if not raw_instances:
        return pd.DataFrame(columns=FEATURES)
    raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
    refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
    raw_area = np.asarray(mask_utils.area(raw_rles), dtype=np.float32).reshape(-1)
    refined_area = np.asarray(mask_utils.area(refined_rles), dtype=np.float32).reshape(-1)
    raw_boxes = np.asarray(mask_utils.toBbox(raw_rles), dtype=np.float32).reshape(-1, 4)
    refined_boxes = np.asarray(mask_utils.toBbox(refined_rles), dtype=np.float32).reshape(-1, 4)
    cross_iou = mask_utils.iou(raw_rles, refined_rles, [0] * len(refined_rles))
    raw_refined_iou = np.diag(cross_iou).astype(np.float32)
    raw_refined_box_iou = _box_iou(raw_boxes, refined_boxes).astype(np.float32)
    rows = []
    for index, refined in enumerate(refined_instances):
        values = refined.get("features", {})
        row = {key: float(values.get(key, 0.0)) for key in BASE_FEATURES}
        row.update({
            "raw_area": float(raw_area[index]),
            "refined_area": float(refined_area[index]),
            "log_area_ratio": float(np.log((refined_area[index] + 1.0) / (raw_area[index] + 1.0))),
            "raw_refined_iou": float(raw_refined_iou[index]),
            "raw_box_fill": float(raw_area[index] / max(raw_boxes[index, 2] * raw_boxes[index, 3], 1.0)),
            "refined_box_fill": float(refined_area[index] / max(refined_boxes[index, 2] * refined_boxes[index, 3], 1.0)),
            "raw_refined_box_iou": float(raw_refined_box_iou[index]),
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def choose_candidate_rles(
    raw_instances: list[dict], refined_instances: list[dict],
    predicted_raw_iou: np.ndarray, predicted_refined_iou: np.ndarray,
    action_mode: str, refine_margin: float,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    if action_mode not in {"raw", "refined", "gated"}:
        raise ValueError(action_mode)
    raw_rles = [rle_from_json(item["rle"]) for item in raw_instances]
    refined_rles = [rle_from_json(item["rle"]) for item in refined_instances]
    raw_score = np.clip(np.asarray(predicted_raw_iou, np.float32), 0.0, 1.0)
    refined_score = np.clip(np.asarray(predicted_refined_iou, np.float32), 0.0, 1.0)
    if action_mode == "raw":
        use_refined = np.zeros(len(raw_rles), dtype=bool); scores = raw_score
    elif action_mode == "refined":
        use_refined = np.ones(len(raw_rles), dtype=bool); scores = refined_score
    else:
        use_refined = refined_score >= raw_score + float(refine_margin)
        scores = np.where(use_refined, refined_score, raw_score)
    chosen = [refined_rles[i] if use_refined[i] else raw_rles[i] for i in range(len(raw_rles))]
    return chosen, scores, use_refined


def cluster_representatives(rles: list[dict], scores: np.ndarray, minimum_score: float, cluster_iou: float) -> list[int]:
    keep = np.flatnonzero(scores >= minimum_score)
    if not len(keep):
        return []
    keep = keep[np.argsort(scores[keep])[::-1]]
    selected: list[int] = []
    for index in keep:
        if not selected:
            selected.append(int(index)); continue
        overlaps = mask_utils.iou([rles[int(index)]], [rles[old] for old in selected], [0] * len(selected))[0]
        if float(overlaps.max(initial=0.0)) < cluster_iou:
            selected.append(int(index))
    return selected


def select_set_aware_instances(
    rles: list[dict], scores: np.ndarray, score_threshold: float,
    min_area: int, cluster_iou: float, duplicate_iou: float,
) -> list[tuple[dict, float]]:
    representatives = cluster_representatives(rles, scores, score_threshold, cluster_iou)
    if not representatives:
        return []
    representative_rles = [rles[index] for index in representatives]
    decoded = mask_utils.decode(representative_rles)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    masks = np.moveaxis(decoded, -1, 0).astype(bool, copy=False)
    representative_scores = np.asarray([scores[index] for index in representatives], np.float32)
    resolved = resolve_instances_scored(masks, representative_scores, score_threshold, min_area, duplicate_iou)
    return [(mask_to_rle_object(mask), score) for mask, score in resolved]
