from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from ..config import add_path_args, add_run_args, discover_data, paths_from_args, resolve_checkpoint
from ..postprocess import mask_to_rle_object, polygon_to_rle, resolve_instances
from ..prediction_cache import PredictionCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Report validation recall/IoU by GT instance size")
    add_path_args(parser)
    add_run_args(parser)
    parser.add_argument("--manifest", "--split-manifest", dest="manifest", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    paths = paths_from_args(args)
    annotation_path, _, _ = discover_data(paths)
    checkpoint = resolve_checkpoint(paths, args.run_name, args.checkpoint)
    output_dir = paths.experiment_results(args.run_name)
    params = json.loads((output_dir / "best_postprocess.json").read_text())
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
    rows = []
    for file_name, group in tqdm(validation.groupby("file_name", sort=True), desc="Size metrics"):
        masks, scores, _ = cache.load(file_name)
        predictions = resolve_instances(masks, scores, float(params["conf"]), int(params["min_area"]))
        pred_rles = [mask_to_rle_object(mask) for mask in predictions]
        for record in group.to_dict("records"):
            image_id = str(record["image_id"])
            info = images[image_id]
            for annotation in annotations[image_id]:
                gt = polygon_to_rle(annotation["segmentation"], info["height"], info["width"])
                best_iou = float(mask_utils.iou(pred_rles, [gt], [0]).max()) if pred_rles else 0.0
                rows.append(
                    {
                        "file_name": file_name,
                        "image_id": image_id,
                        "annotation_id": annotation["id"],
                        "area": float(annotation.get("area", mask_utils.area(gt))),
                        "best_iou": best_iou,
                        "matched": best_iou > 0.5,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["size_bin"] = pd.qcut(frame["area"].rank(method="first"), q=3, labels=["small", "medium", "large"])
    summary = frame.groupby("size_bin", observed=True).agg(
        gt_instances=("annotation_id", "size"),
        recall=("matched", "mean"),
        mean_best_iou=("best_iou", "mean"),
        median_area=("area", "median"),
    ).reset_index()
    frame.to_csv(output_dir / "metrics_by_gt_instance.csv", index=False)
    summary.to_csv(output_dir / "metrics_by_size.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
