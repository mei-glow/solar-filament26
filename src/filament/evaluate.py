from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .config import add_config_arg, add_path_args, add_run_args, discover_data, parse_args_with_config, paths_from_args, resolve_checkpoint
from .experiment import record_event, sha256_file, write_json
from .postprocess import mask_to_rle_object, polygon_to_rle, pq_stats_rle, resolve_instances_scored
from .prediction_cache import PredictionCache


def _numbers(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune post-processing using grouped validation PQ")
    add_path_args(parser)
    add_run_args(parser)
    add_config_arg(parser)
    parser.add_argument("--manifest", "--split-manifest", dest="manifest", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-values", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--area-values", default="25,50,100,200")
    parser.add_argument("--quick", action="store_true", help="Use a 6-setting coarse grid")
    parser.add_argument("--no-reuse-cache", action="store_true")
    parser.add_argument(
        "--selection-metric", choices=("global_pq", "mean_pq"), default="global_pq",
        help="Metric used for best_postprocess.json; official Kaggle PQ is pooled/global.",
    )
    args = parse_args_with_config(parser)

    paths = paths_from_args(args)
    annotation_path, train_image_dir, _ = discover_data(paths)
    checkpoint = resolve_checkpoint(paths, args.run_name, args.checkpoint)
    manifest_path = args.manifest.resolve() if args.manifest else paths.results_root / "split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest or "image_id" not in manifest:
        raise ValueError(f"Manifest must contain split and image_id: {manifest_path}")
    validation = manifest[manifest["split"] == "val"].copy()
    if validation.empty:
        raise ValueError(f"No validation records in {manifest_path}")

    with annotation_path.open() as handle:
        coco = json.load(handle)
    images_by_id = {str(image["id"]): image for image in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    gt_by_id = {}
    for image_id in validation["image_id"].astype(str):
        image = images_by_id[image_id]
        gt_by_id[image_id] = [
            polygon_to_rle(annotation["segmentation"], image["height"], image["width"])
            for annotation in annotations[image_id]
        ]

    if args.quick:
        conf_values, area_values = (0.20, 0.30, 0.40), (50, 100)
    else:
        conf_values = _numbers(args.conf_values, float)
        area_values = _numbers(args.area_values, int)
    configurations = [(conf, area) for conf in conf_values for area in area_values]
    stats = {
        config: {"pq_sum": 0.0, "sq_sum": 0.0, "rq_sum": 0.0, "iou_sum": 0.0, "count": 0, "tp": 0, "fp": 0, "fn": 0}
        for config in configurations
    }
    by_image_rows: list[dict] = []
    records_by_file = {
        file_name: group.to_dict("records")
        for file_name, group in validation.groupby("file_name", sort=True)
    }

    cache = PredictionCache(
        paths.prediction_cache(args.run_name),
        checkpoint,
        {"imgsz": args.imgsz, "base_conf": args.base_conf, "nms_iou": args.nms_iou, "max_det": args.max_det},
        reuse=not args.no_reuse_cache,
    )
    model = None
    for file_name, records in tqdm(records_by_file.items(), desc="Inference/cache + PQ grid"):
        if cache.has(file_name) and not args.no_reuse_cache:
            masks, confidences, _ = cache.load(file_name)
        else:
            if model is None:
                model = YOLO(str(checkpoint))
            result = model.predict(
                str(train_image_dir / file_name), imgsz=args.imgsz, conf=args.base_conf,
                iou=args.nms_iou, max_det=args.max_det, retina_masks=True,
                device=args.device, verbose=False,
            )[0]
            if result.masks is None:
                masks = np.empty((0, 1, 1), bool)
                confidences = np.empty(0, np.float32)
                boxes = np.empty((0, 4), np.float32)
            else:
                masks = result.masks.data.cpu().numpy().astype(bool)
                confidences = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
            cache.save(file_name, masks, confidences, boxes)

        resolved_by_area = {}
        for min_area in area_values:
            resolved = resolve_instances_scored(masks, confidences, conf=min(conf_values), min_area=min_area)
            resolved_by_area[min_area] = [(mask_to_rle_object(mask), score) for mask, score in resolved]

        for conf, min_area in configurations:
            pred_rles = [rle for rle, score in resolved_by_area[min_area] if score >= conf]
            current = stats[(conf, min_area)]
            for record in records:
                image_id = str(record["image_id"])
                values = pq_stats_rle(gt_by_id[image_id], pred_rles)
                current["pq_sum"] += values["pq"]
                current["sq_sum"] += values["sq"]
                current["rq_sum"] += values["rq"]
                current["iou_sum"] += values["iou_sum"]
                current["tp"] += values["tp"]
                current["fp"] += values["fp"]
                current["fn"] += values["fn"]
                current["count"] += 1
                by_image_rows.append(
                    {"file_name": file_name, "image_id": image_id, "conf": conf, "min_area": min_area, **values}
                )

    rows = []
    for (conf, min_area), values in stats.items():
        denominator = values["tp"] + 0.5 * values["fp"] + 0.5 * values["fn"]
        rows.append(
            {
                "conf": conf,
                "min_area": min_area,
                "mean_pq": values["pq_sum"] / values["count"],
                "mean_sq": values["sq_sum"] / values["count"],
                "mean_rq": values["rq_sum"] / values["count"],
                "global_pq": values["iou_sum"] / denominator if denominator else 0.0,
                "tp": values["tp"], "fp": values["fp"], "fn": values["fn"],
            }
        )
    tuning = pd.DataFrame(rows).sort_values(args.selection_metric, ascending=False)
    output_dir = paths.experiment_results(args.run_name)
    tuning.to_csv(output_dir / "postprocess_tuning.csv", index=False)
    pd.DataFrame(by_image_rows).to_csv(output_dir / "metrics_by_annotation.csv", index=False)
    best = tuning.iloc[0].to_dict()
    best["selection_metric"] = args.selection_metric
    write_json(output_dir / "best_postprocess.json", best)
    record_event(
        paths.artifact_root,
        args.run_name,
        "evaluate",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "best": best,
        },
    )
    print(tuning.head(12).to_string(index=False))
    print(f"Saved: {output_dir / 'best_postprocess.json'}")


if __name__ == "__main__":
    main()
