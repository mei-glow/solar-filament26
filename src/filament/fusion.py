from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.segmentation import watershed


@dataclass
class Candidate:
    mask: np.ndarray
    score: float
    support: int = 1


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    if not intersection:
        return 0.0
    return float(intersection / np.logical_or(left, right).sum())


def cluster_detector_candidates(
    predictions: list[tuple[np.ndarray, np.ndarray]],
    confidence: float,
    cluster_iou: float = 0.45,
) -> list[Candidate]:
    raw = []
    for masks, scores in predictions:
        for mask, score in zip(masks, scores):
            if float(score) >= confidence and mask.any():
                raw.append(Candidate(mask.astype(bool), float(score)))
    raw.sort(key=lambda item: item.score, reverse=True)
    clusters: list[Candidate] = []
    for candidate in raw:
        best_index, best_iou = -1, 0.0
        for index, cluster in enumerate(clusters):
            current = mask_iou(candidate.mask, cluster.mask)
            if current > best_iou:
                best_index, best_iou = index, current
        if best_iou >= cluster_iou:
            cluster = clusters[best_index]
            cluster.mask = np.logical_or(cluster.mask, candidate.mask)
            cluster.score = max(cluster.score, candidate.score)
            cluster.support += 1
        else:
            clusters.append(candidate)
    return clusters


def _expanded_bounds(mask: np.ndarray, expansion: float) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x = int(round((x1 - x0) * expansion))
    pad_y = int(round((y1 - y0) * expansion))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(mask.shape[1], x1 + pad_x), min(mask.shape[0], y1 + pad_y)


def _seed(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    seed = cv2.erode(mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    if seed.sum() < 4:
        seed = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return seed if seed.any() else mask.copy()


def _component_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    overlaps = np.bincount(labels[seed].ravel(), minlength=count)
    overlaps[0] = 0
    label = int(np.argmax(overlaps))
    return labels == label if overlaps[label] else np.zeros_like(mask)


def fuse_instances(
    semantic_output: np.ndarray,
    detector_predictions: list[tuple[np.ndarray, np.ndarray]],
    detector_confidence: float,
    semantic_threshold: float,
    min_area: int,
    expansion: float = 0.20,
    cluster_iou: float = 0.45,
    centerline_boost: float = 0.10,
    rescue_threshold: float | None = None,
    rescue_elongation: float = 1.5,
) -> list[tuple[np.ndarray, float]]:
    probability = np.clip(
        semantic_output[0] + centerline_boost * semantic_output[1], 0.0, 1.0
    )
    clusters = cluster_detector_candidates(detector_predictions, detector_confidence, cluster_iou)
    if not clusters:
        clusters = []
    markers = np.zeros(probability.shape, dtype=np.int32)
    occupied_seed = np.zeros(probability.shape, dtype=bool)
    retained: list[tuple[Candidate, np.ndarray, tuple[int, int, int, int], int]] = []
    for label, candidate in enumerate(clusters, start=1):
        seed = np.logical_and(_seed(candidate.mask), ~occupied_seed)
        if not seed.any():
            seed = np.logical_and(candidate.mask, ~occupied_seed)
        if not seed.any():
            continue
        markers[seed] = label
        occupied_seed |= seed
        retained.append((candidate, seed, _expanded_bounds(candidate.mask, expansion), label))

    foreground = probability >= semantic_threshold
    foreground |= occupied_seed
    labels = watershed(-probability, markers=markers, mask=foreground, connectivity=2) if retained else np.zeros_like(markers)
    instances: list[tuple[np.ndarray, float]] = []
    occupied = np.zeros(probability.shape, dtype=bool)
    for candidate, seed, (x0, y0, x1, y1), label in retained:
        region = labels == label
        clipped = np.zeros_like(region)
        clipped[y0:y1, x0:x1] = region[y0:y1, x0:x1]
        clipped = _component_touching_seed(clipped, seed)
        if clipped.sum() < min_area:
            fallback = np.logical_and(candidate.mask, ~occupied)
            clipped = fallback if fallback.sum() >= min_area else clipped
        clipped &= ~occupied
        if clipped.sum() >= min_area:
            instances.append((clipped, candidate.score))
            occupied |= clipped

    if rescue_threshold is not None:
        unassigned = (probability >= rescue_threshold) & ~occupied & (labels == 0)
        count, components, stats, _ = cv2.connectedComponentsWithStats(unassigned.astype(np.uint8), 8)
        rescued = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            elongation = max(width, height) / max(min(width, height), 1)
            component = components == label
            max_probability = float(probability[component].max())
            if elongation >= rescue_elongation and max_probability >= rescue_threshold:
                rescued.append((component, max_probability))
        rescued.sort(key=lambda item: item[1], reverse=True)
        for component, score in rescued:
            component &= ~occupied
            if component.sum() >= min_area:
                instances.append((component, score))
                occupied |= component
    return instances
