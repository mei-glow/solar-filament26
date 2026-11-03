from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils
def polygon_to_mask(polygons, height: int = 2048, width: int = 2048) -> np.ndarray:
    return mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))).astype(bool)


def polygon_to_rle(polygons, height: int = 2048, width: int = 2048) -> dict:
    return mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))


def mask_to_rle_object(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def mask_to_rle(mask: np.ndarray) -> str:
    encoded = mask_to_rle_object(mask)
    return encoded["counts"].decode("utf-8")


def pairwise_iou(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    gt = np.stack(gt_masks).reshape(len(gt_masks), -1)
    pred = np.stack(pred_masks).reshape(len(pred_masks), -1)
    intersection = gt.astype(np.uint32) @ pred.astype(np.uint32).T
    union = gt.sum(1)[:, None] + pred.sum(1)[None, :] - intersection
    return intersection / np.maximum(union, 1)


def pq_one(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], threshold: float = 0.5):
    ious = pairwise_iou(gt_masks, pred_masks)
    if ious.size:
        hits = ious > threshold
        tp = int(hits.sum())
        iou_sum = float(ious[hits].sum())
        fp = int((hits.sum(axis=0) == 0).sum())
        fn = int((hits.sum(axis=1) == 0).sum())
    else:
        tp, iou_sum = 0, 0.0
        fp, fn = len(pred_masks), len(gt_masks)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return (iou_sum / denominator if denominator else 0.0), tp, fp, fn


def pq_one_rle(gt_rles: list[dict], pred_rles: list[dict], threshold: float = 0.5):
    stats = pq_stats_rle(gt_rles, pred_rles, threshold)
    return stats["pq"], stats["tp"], stats["fp"], stats["fn"]


def pq_stats_rle(gt_rles: list[dict], pred_rles: list[dict], threshold: float = 0.5) -> dict:
    if gt_rles and pred_rles:
        # pycocotools returns rows=detections and columns=ground truth.
        ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T
        hits = ious > threshold
        tp = int(hits.sum())
        iou_sum = float(ious[hits].sum())
        fp = int((hits.sum(axis=0) == 0).sum())
        fn = int((hits.sum(axis=1) == 0).sum())
    else:
        tp, iou_sum = 0, 0.0
        fp, fn = len(pred_rles), len(gt_rles)
    denominator = tp + 0.5 * fp + 0.5 * fn
    pq = iou_sum / denominator if denominator else 0.0
    sq = iou_sum / tp if tp else 0.0
    rq = tp / denominator if denominator else 0.0
    return {"pq": pq, "sq": sq, "rq": rq, "iou_sum": iou_sum, "tp": tp, "fp": fp, "fn": fn}


def rle_to_json(rle: dict) -> dict:
    counts = rle["counts"]
    return {"size": list(rle["size"]), "counts": counts.decode("utf-8") if isinstance(counts, bytes) else counts}


def rle_from_json(payload: dict) -> dict:
    counts = payload["counts"]
    return {"size": payload["size"], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}


def resolve_instances(
    masks: np.ndarray,
    scores: np.ndarray,
    conf: float = 0.30,
    min_area: int = 25,
    duplicate_iou: float = 0.80,
) -> list[np.ndarray]:
    return [mask for mask, _ in resolve_instances_scored(masks, scores, conf, min_area, duplicate_iou)]


def resolve_instances_scored(
    masks: np.ndarray,
    scores: np.ndarray,
    conf: float = 0.30,
    min_area: int = 25,
    duplicate_iou: float = 0.80,
) -> list[tuple[np.ndarray, float]]:
    """Resolve masks once while retaining confidence for cheap threshold sweeps.

    Since candidates are processed from high to low confidence, a lower-score
    candidate never changes an already accepted higher-score mask. Therefore a
    result resolved at the minimum confidence can be filtered at any higher
    confidence without repeating full-resolution overlap operations.
    """
    keep = np.flatnonzero(scores >= conf)
    keep = keep[np.argsort(scores[keep])[::-1]]
    accepted: list[tuple[np.ndarray, float]] = []
    occupied = None
    for idx in keep:
        mask = masks[idx].astype(bool)
        if mask.sum() < min_area:
            continue
        if any(
            (lambda union, inter: bool(union and inter / union >= duplicate_iou))(
                np.logical_or(mask, old).sum(), np.logical_and(mask, old).sum()
            )
            for old, _ in accepted
        ):
            continue
        if occupied is not None:
            mask = np.logical_and(mask, ~occupied)
        if mask.sum() < min_area:
            continue
        accepted.append((mask, float(scores[idx])))
        occupied = mask.copy() if occupied is None else np.logical_or(occupied, mask)
    return accepted

