from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm
from ultralytics import YOLO

from .config import add_path_args, discover_data, paths_from_args
from .postprocess import mask_to_rle_object, polygon_to_mask, pq_stats_rle
from .prediction_cache import PredictionCache


def _box_iou(gt: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float32)
    x, y, w, h = gt.astype(float)
    left = np.maximum(x, boxes[:, 0]); top = np.maximum(y, boxes[:, 1])
    right = np.minimum(x + w, boxes[:, 2]); bottom = np.minimum(y + h, boxes[:, 3])
    inter = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = w * h + np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def _greedy_oracle(gt_masks: list[np.ndarray], proposals: list[np.ndarray]) -> list[np.ndarray]:
    if not gt_masks or not proposals:
        return []
    gt_rles = [mask_to_rle_object(mask) for mask in gt_masks]
    pred_rles = [mask_to_rle_object(mask) for mask in proposals]
    ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T
    pairs = [(float(ious[g, p]), g, p) for g in range(len(gt_masks)) for p in range(len(proposals))]
    pairs.sort(reverse=True)
    used_gt, used_pred, selected = set(), set(), []
    for iou, gt_index, pred_index in pairs:
        # An oracle never exports a proposal that cannot score as a TP.
        if iou <= 0.5 or gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index); used_pred.add(pred_index); selected.append(proposals[pred_index])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 proposal coverage and oracle ceiling audit")
    add_path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", default="p1_proposal_oracle")
    parser.add_argument("--detector-run-names", required=True)
    parser.add_argument("--detector-checkpoints", required=True)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--coverage-mask-iou", type=float, default=0.10)
    args = parser.parse_args()

    paths = paths_from_args(args)
    annotation_path, train_image_dir, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve())
    validation = manifest[manifest["split"] == "val"].copy() if "split" in manifest else manifest[manifest["fold"] == 0].copy()
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in coco["annotations"]:
        annotations[str(item["image_id"])].append(item)
    names = [item.strip() for item in args.detector_run_names.split(",")]
    checkpoints = [Path(item.strip()).resolve() for item in args.detector_checkpoints.split(",")]
    if len(names) != len(checkpoints):
        raise ValueError("detector names/checkpoints length mismatch")
    caches = [PredictionCache(paths.prediction_cache(name), ckpt, {
        "imgsz": args.imgsz, "base_conf": args.base_conf, "nms_iou": args.nms_iou, "max_det": args.max_det,
    }) for name, ckpt in zip(names, checkpoints)]
    models: list[YOLO | None] = [None] * len(caches)

    rows, totals = [], {"existing": dict(iou_sum=0.0, tp=0, fp=0, fn=0), "perfect": dict(iou_sum=0.0, tp=0, fp=0, fn=0)}
    for file_name, records in tqdm(validation.groupby("file_name", sort=True), desc="Proposal oracle"):
        proposals, boxes, scores, sources = [], [], [], []
        for source_index, (name, cache) in enumerate(zip(names, caches)):
            if not cache.has(file_name):
                if models[source_index] is None:
                    models[source_index] = YOLO(str(checkpoints[source_index]))
                result = models[source_index].predict(
                    str(train_image_dir / file_name), imgsz=args.imgsz,
                    conf=args.base_conf, iou=args.nms_iou, max_det=args.max_det,
                    retina_masks=True, device=args.device, verbose=False,
                )[0]
                if result.masks is None:
                    inferred_masks = np.empty((0, 1, 1), dtype=bool)
                    inferred_scores = np.empty(0, dtype=np.float32)
                    inferred_boxes = np.empty((0, 4), dtype=np.float32)
                else:
                    inferred_masks = result.masks.data.cpu().numpy().astype(bool)
                    inferred_scores = result.boxes.conf.cpu().numpy()
                    inferred_boxes = result.boxes.xyxy.cpu().numpy()
                cache.save(file_name, inferred_masks, inferred_scores, inferred_boxes)
            masks, conf, current_boxes = cache.load(file_name)
            proposals.extend(list(masks)); boxes.extend(list(current_boxes)); scores.extend(list(conf)); sources.extend([name] * len(conf))
        boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        pred_rles = [mask_to_rle_object(mask) for mask in proposals]
        for record in records.itertuples(index=False):
            image_id = str(record.image_id); info = images[image_id]; anns = annotations[image_id]
            gt_masks = [polygon_to_mask(item["segmentation"], info["height"], info["width"]) for item in anns]
            gt_rles = [mask_to_rle_object(mask) for mask in gt_masks]
            ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T if pred_rles and gt_rles else np.zeros((len(gt_rles), len(pred_rles)))
            selected = _greedy_oracle(gt_masks, proposals)
            existing = pq_stats_rle(gt_rles, [mask_to_rle_object(mask) for mask in selected])
            covered_gt = [mask for index, mask in enumerate(gt_masks) if ious.shape[1] and float(ious[index].max()) > args.coverage_mask_iou]
            perfect = pq_stats_rle(gt_rles, [mask_to_rle_object(mask) for mask in covered_gt])
            for key, values in (("existing", existing), ("perfect", perfect)):
                for metric in totals[key]: totals[key][metric] += values[metric]
            for gt_index, (ann, mask) in enumerate(zip(anns, gt_masks)):
                mask_ious = ious[gt_index] if ious.shape[1] else np.empty(0)
                current_box_ious = _box_iou(np.asarray(ann["bbox"]), boxes_array)
                best_index = int(np.argmax(mask_ious)) if len(mask_ious) else -1
                x, y, width, height = map(float, ann["bbox"])
                cx, cy = x + width / 2, y + height / 2
                radius = np.hypot(cx - info["width"] / 2, cy - info["height"] / 2) / (min(info["width"], info["height"]) / 2)
                area = float(mask.sum()); perimeter = float(cv2_arc(mask))
                rows.append({
                    "file_name": file_name, "image_id": image_id, "annotation_id": ann.get("id"),
                    "area": area, "thickness_proxy": area / max(perimeter, 1.0), "elongation": max(width, height) / max(min(width, height), 1.0),
                    "radial_position": radius, "best_mask_iou": float(mask_ious.max()) if len(mask_ious) else 0.0,
                    "best_box_iou": float(current_box_ious.max()) if len(current_box_ious) else 0.0,
                    "best_score": float(scores[best_index]) if best_index >= 0 else 0.0,
                    "best_source": sources[best_index] if best_index >= 0 else "none",
                    "proposal_count_mask_iou_01": int((mask_ious > .1).sum()),
                    "proposal_count_mask_iou_05": int((mask_ious > .5).sum()),
                    "merge_risk": int(any((ious[:, p] > .1).sum() > 1 for p in range(ious.shape[1]))),
                })

    frame = pd.DataFrame(rows)
    report = {"instances": len(frame), "annotation_records": len(validation)}
    for kind in ("mask", "box"):
        for threshold in (.1, .3, .5):
            report[f"{kind}_recall_{threshold}"] = float((frame[f"best_{kind}_iou"] > threshold).mean())
    for key, values in totals.items():
        denominator = values["tp"] + .5 * values["fp"] + .5 * values["fn"]
        report[f"{key}_oracle_global_pq"] = values["iou_sum"] / denominator if denominator else 0.0
        report[f"{key}_oracle_tp"] = values["tp"]; report[f"{key}_oracle_fp"] = values["fp"]; report[f"{key}_oracle_fn"] = values["fn"]
    output = paths.experiment_results(args.run_name)
    frame.to_csv(output / "proposal_oracle_instances.csv", index=False)
    (output / "proposal_oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); print(f"Saved: {output}")


def cv2_arc(mask: np.ndarray) -> float:
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(contour, True) for contour in contours)


if __name__ == "__main__":
    main()
