from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..config import add_path_args, add_run_args, discover_data, paths_from_args, resolve_checkpoint
from ..postprocess import polygon_to_mask, resolve_instances
from ..prediction_cache import PredictionCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a gallery of the worst validation predictions")
    add_path_args(parser)
    add_run_args(parser)
    parser.add_argument("--manifest", "--split-manifest", dest="manifest", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    paths = paths_from_args(args)
    annotation_path, image_dir, _ = discover_data(paths)
    checkpoint = resolve_checkpoint(paths, args.run_name, args.checkpoint)
    output_dir = paths.experiment_results(args.run_name)
    params = json.loads((output_dir / "best_postprocess.json").read_text())
    metrics = pd.read_csv(output_dir / "metrics_by_annotation.csv")
    selected = metrics[
        np.isclose(metrics["conf"], float(params["conf"]))
        & (metrics["min_area"] == int(params["min_area"]))
    ]
    worst_files = selected.groupby("file_name")["pq"].mean().nsmallest(args.count)
    manifest_path = args.manifest.resolve() if args.manifest else paths.results_root / "split_manifest.csv"
    validation = pd.read_csv(manifest_path).query("split == 'val'")

    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(item["id"]): item for item in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    cache = PredictionCache(
        paths.prediction_cache(args.run_name), checkpoint,
        {"imgsz": args.imgsz, "base_conf": args.base_conf, "nms_iou": args.nms_iou, "max_det": args.max_det},
    )

    tiles = []
    for file_name, pq in worst_files.items():
        record = validation[validation["file_name"] == file_name].iloc[0]
        image_id = str(record["image_id"])
        info = images[image_id]
        gray = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gt_masks = [polygon_to_mask(item["segmentation"], info["height"], info["width"]) for item in annotations[image_id]]
        gt = np.logical_or.reduce(gt_masks) if gt_masks else np.zeros_like(gray, dtype=bool)
        masks, scores, _ = cache.load(file_name)
        predictions = resolve_instances(masks, scores, float(params["conf"]), int(params["min_area"]))
        pred = np.logical_or.reduce(predictions) if predictions else np.zeros_like(gray, dtype=bool)
        overlay = rgb.copy()
        overlay[gt & pred] = (0, 210, 0)
        overlay[gt & ~pred] = (0, 220, 255)
        overlay[~gt & pred] = (0, 0, 255)
        blended = cv2.addWeighted(rgb, 0.45, overlay, 0.55, 0)
        tile = cv2.resize(blended, (384, 384), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{Path(file_name).stem} PQ={pq:.3f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        raise RuntimeError("No validation images selected")
    columns = 5
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row += [blank] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    gallery = np.concatenate(rows, axis=0)
    target = output_dir / "worst_predictions_gallery.jpg"
    cv2.imwrite(str(target), gallery, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Green=overlap, yellow=FN, red=FP. Saved: {target}")


if __name__ == "__main__":
    main()
