from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ..config import add_path_args, add_run_args, discover_data, paths_from_args, resolve_checkpoint
from ..experiment import record_event, write_json
from ..postprocess import mask_to_rle_object, polygon_to_rle, resolve_instances
from ..prediction_cache import PredictionCache


def _box_iou(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float32)
    gt = np.asarray([mask_utils.toBbox(rle) for rle in gt_rles], dtype=np.float32)
    pred = np.asarray([mask_utils.toBbox(rle) for rle in pred_rles], dtype=np.float32)
    gx1, gy1, gw, gh = gt.T
    px1, py1, pw, ph = pred.T
    gx2, gy2, px2, py2 = gx1 + gw, gy1 + gh, px1 + pw, py1 + ph
    inter_w = np.maximum(0, np.minimum(gx2[:, None], px2[None]) - np.maximum(gx1[:, None], px1[None]))
    inter_h = np.maximum(0, np.minimum(gy2[:, None], py2[None]) - np.maximum(gy1[:, None], py1[None]))
    inter = inter_w * inter_h
    union = (gw * gh)[:, None] + (pw * ph)[None] - inter
    return inter / np.maximum(union, 1e-6)


def classify_record(gt_rles: list[dict], pred_rles: list[dict]) -> dict:
    if gt_rles and pred_rles:
        ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T
        box_ious = _box_iou(gt_rles, pred_rles)
        rows, cols = linear_sum_assignment(-ious)
        matched = [(int(r), int(c)) for r, c in zip(rows, cols) if ious[r, c] > 0.5]
    else:
        ious = np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float32)
        box_ious = np.zeros_like(ious)
        matched = []
    matched_gt = {r for r, _ in matched}
    matched_pred = {c for _, c in matched}
    counts = Counter(tp=len(matched))
    for gt_index in range(len(gt_rles)):
        if gt_index in matched_gt:
            continue
        candidates = int((ious[gt_index] > 0.10).sum()) if len(pred_rles) else 0
        max_iou = float(ious[gt_index].max()) if len(pred_rles) else 0.0
        max_box_iou = float(box_ious[gt_index].max()) if len(pred_rles) else 0.0
        if candidates >= 2:
            counts["split_gt"] += 1
        elif max_iou > 0.05 or max_box_iou > 0.10:
            counts["mask_iou_failure"] += 1
        else:
            counts["missed_detection"] += 1
    for pred_index in range(len(pred_rles)):
        if pred_index in matched_pred:
            continue
        gt_support = int((ious[:, pred_index] > 0.10).sum()) if len(gt_rles) else 0
        counts["merge_prediction" if gt_support >= 2 else "false_positive"] += 1
    counts["gt"] = len(gt_rles)
    counts["pred"] = len(pred_rles)
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify validation errors using cached predictions")
    add_path_args(parser)
    add_run_args(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--postprocess", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    paths = paths_from_args(args)
    annotation_path, _, _ = discover_data(paths)
    checkpoint = resolve_checkpoint(paths, args.run_name, args.checkpoint)
    manifest_path = args.manifest.resolve() if args.manifest else paths.results_root / "split_manifest.csv"
    validation = pd.read_csv(manifest_path).query("split == 'val'")
    output_dir = paths.experiment_results(args.run_name)
    postprocess_path = args.postprocess.resolve() if args.postprocess else output_dir / "best_postprocess.json"
    params = json.loads(postprocess_path.read_text())

    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(image["id"]): image for image in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    cache = PredictionCache(
        paths.prediction_cache(args.run_name), checkpoint,
        {"imgsz": args.imgsz, "base_conf": args.base_conf, "nms_iou": args.nms_iou, "max_det": args.max_det},
    )

    rows = []
    for file_name, group in tqdm(validation.groupby("file_name", sort=True), desc="Error taxonomy"):
        if not cache.has(file_name):
            raise FileNotFoundError(f"Missing cache for {file_name}; run filament.evaluate first")
        masks, scores, _ = cache.load(file_name)
        resolved = resolve_instances(masks, scores, conf=float(params["conf"]), min_area=int(params["min_area"]))
        pred_rles = [mask_to_rle_object(mask) for mask in resolved]
        for record in group.to_dict("records"):
            image_id = str(record["image_id"])
            image = images[image_id]
            gt_rles = [polygon_to_rle(a["segmentation"], image["height"], image["width"]) for a in annotations[image_id]]
            rows.append({"file_name": file_name, "image_id": image_id, **classify_record(gt_rles, pred_rles)})
    frame = pd.DataFrame(rows).fillna(0)
    frame.to_csv(output_dir / "error_taxonomy_by_annotation.csv", index=False)
    numeric = [c for c in frame.columns if c not in ("file_name", "image_id")]
    summary = {key: int(frame[key].sum()) for key in numeric}
    write_json(output_dir / "error_taxonomy_summary.json", summary)
    record_event(paths.artifact_root, args.run_name, "error_taxonomy", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
