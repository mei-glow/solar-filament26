from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import record_event, sha256_file, write_json
from .fusion import fuse_instances
from .postprocess import mask_to_rle_object, polygon_to_rle, pq_stats_rle
from .prediction_cache import PredictionCache
from .semantic_inference import SemanticPredictionCache, infer_semantic_image, load_semantic_checkpoint


def _values(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def _rescue_values(text: str) -> list[float | None]:
    return [
        None if item.strip().lower() in ("none", "off") else float(item.strip())
        for item in text.split(",") if item.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dual-branch instance/topology fusion with official PQ")
    add_path_args(parser)
    parser.add_argument("--run-name", required=True, help="Output fusion experiment name")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-run-name", required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, default=None)
    parser.add_argument("--detector-run-names", default="p2_d3_yolov8l_2048_conservative,p2_d4_yolov8l_2048_medoid")
    parser.add_argument("--detector-checkpoints", required=True, help="Comma-separated checkpoints matching detector run names")
    parser.add_argument("--detector-imgsz", type=int, default=2048)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.50)
    parser.add_argument("--semantic-batch", type=int, default=4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--detector-conf-values", default="0.20,0.25,0.30")
    parser.add_argument("--semantic-threshold-values", default="0.35,0.45,0.55")
    parser.add_argument("--min-area-values", default="100,200")
    parser.add_argument("--rescue-threshold-values", default="none")
    parser.add_argument("--expansion", type=float, default=0.20)
    parser.add_argument("--cluster-iou", type=float, default=0.45)
    parser.add_argument("--centerline-boost", type=float, default=None, help="Default: 0.10 only when the centerline head was trained")
    parser.add_argument(
        "--selection-metric", choices=("global_pq", "mean_pq"), default="global_pq",
        help="Metric used to promote best_fusion.json; official Kaggle PQ is pooled/global.",
    )
    args = parser.parse_args()

    paths = paths_from_args(args)
    annotation_path, train_image_dir, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve())
    validation = manifest[manifest["split"] == "val"].copy()
    records_by_file = {name: group.to_dict("records") for name, group in validation.groupby("file_name", sort=True)}
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(image["id"]): image for image in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    gt_by_id = {
        str(record.image_id): [
            polygon_to_rle(item["segmentation"], images[str(record.image_id)]["height"], images[str(record.image_id)]["width"])
            for item in annotations[str(record.image_id)]
        ]
        for record in validation.itertuples(index=False)
    }

    semantic_checkpoint = args.semantic_checkpoint.resolve() if args.semantic_checkpoint else paths.run_dir(args.semantic_run_name) / "best.pt"
    if not semantic_checkpoint.is_file():
        raise FileNotFoundError(semantic_checkpoint)
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    semantic_model, semantic_config = load_semantic_checkpoint(semantic_checkpoint, device)
    tile_size = args.tile_size or int(semantic_config["tile_size"])
    feature_mode = semantic_config["feature_mode"]
    centerline_boost = args.centerline_boost
    if centerline_boost is None:
        centerline_boost = 0.10 if float(semantic_config.get("centerline_weight", 0.0)) > 0 else 0.0
    semantic_cache_key = f"{args.semantic_run_name}_t{tile_size}_o{int(round(args.overlap * 100))}_tta{int(args.tta)}"
    semantic_cache = SemanticPredictionCache(
        paths.artifact_root / "predictions" / "semantic" / semantic_cache_key / "val",
        semantic_checkpoint,
        {"tile_size": tile_size, "overlap": args.overlap, "feature_mode": feature_mode, "tta": args.tta},
    )

    detector_names = _values(args.detector_run_names, str)
    detector_checkpoints = [Path(item).expanduser().resolve() for item in _values(args.detector_checkpoints, str)]
    if len(detector_names) != len(detector_checkpoints):
        raise ValueError("detector run names/checkpoints length mismatch")
    detector_caches = [
        PredictionCache(
            paths.prediction_cache(name), checkpoint,
            {"imgsz": args.detector_imgsz, "base_conf": args.base_conf, "nms_iou": args.nms_iou, "max_det": args.max_det},
        )
        for name, checkpoint in zip(detector_names, detector_checkpoints)
    ]
    configurations = [
        (detector_conf, semantic_threshold, min_area, rescue_threshold)
        for detector_conf in _values(args.detector_conf_values, float)
        for semantic_threshold in _values(args.semantic_threshold_values, float)
        for min_area in _values(args.min_area_values, int)
        for rescue_threshold in _rescue_values(args.rescue_threshold_values)
    ]
    stats = {
        config: {"pq": 0.0, "sq": 0.0, "rq": 0.0, "iou_sum": 0.0, "count": 0, "tp": 0, "fp": 0, "fn": 0}
        for config in configurations
    }

    for file_name, records in tqdm(records_by_file.items(), desc="Semantic cache + fusion PQ"):
        if semantic_cache.has(file_name):
            semantic_output = semantic_cache.load(file_name)
        else:
            image = cv2.imread(str(train_image_dir / file_name), cv2.IMREAD_GRAYSCALE)
            semantic_output = infer_semantic_image(
                semantic_model, image, device, tile_size, args.overlap, feature_mode,
                batch_size=args.semantic_batch, tta=args.tta,
            )
            semantic_cache.save(file_name, semantic_output)
        detector_predictions = []
        for cache in detector_caches:
            if not cache.has(file_name):
                raise FileNotFoundError(f"Detector cache missing {file_name} in {cache.root}; run filament.evaluate first")
            masks, scores, _ = cache.load(file_name)
            detector_predictions.append((masks, scores))
        for config in configurations:
            detector_conf, semantic_threshold, min_area, rescue_threshold = config
            instances = fuse_instances(
                semantic_output, detector_predictions, detector_conf, semantic_threshold, min_area,
                expansion=args.expansion, cluster_iou=args.cluster_iou,
                centerline_boost=centerline_boost, rescue_threshold=rescue_threshold,
            )
            pred_rles = [mask_to_rle_object(mask) for mask, _ in instances]
            current = stats[config]
            for record in records:
                values = pq_stats_rle(gt_by_id[str(record["image_id"])], pred_rles)
                for key in ("pq", "sq", "rq", "iou_sum", "tp", "fp", "fn"):
                    current[key] += values[key]
                current["count"] += 1

    rows = []
    for (detector_conf, semantic_threshold, min_area, rescue_threshold), values in stats.items():
        denominator = values["tp"] + 0.5 * values["fp"] + 0.5 * values["fn"]
        rows.append({
            "detector_conf": detector_conf,
            "semantic_threshold": semantic_threshold,
            "min_area": min_area,
            "rescue_threshold": rescue_threshold,
            "mean_pq": values["pq"] / values["count"],
            "mean_sq": values["sq"] / values["count"],
            "mean_rq": values["rq"] / values["count"],
            "global_pq": values["iou_sum"] / denominator if denominator else 0.0,
            "tp": int(values["tp"]), "fp": int(values["fp"]), "fn": int(values["fn"]),
        })
    ranking = pd.DataFrame(rows).sort_values(args.selection_metric, ascending=False)
    output_dir = paths.experiment_results(args.run_name)
    ranking.to_csv(output_dir / "fusion_tuning.csv", index=False)
    best = {key: (None if pd.isna(value) else value) for key, value in ranking.iloc[0].to_dict().items()}
    best.update({
        "semantic_run_name": args.semantic_run_name,
        "semantic_cache_key": semantic_cache_key,
        "semantic_checkpoint": str(semantic_checkpoint),
        "detector_run_names": detector_names,
        "detector_checkpoints": [str(path) for path in detector_checkpoints],
        "tile_size": tile_size, "overlap": args.overlap, "feature_mode": feature_mode,
        "expansion": args.expansion, "cluster_iou": args.cluster_iou,
        "centerline_boost": centerline_boost, "tta": args.tta,
        "selection_metric": args.selection_metric,
    })
    write_json(output_dir / "best_fusion.json", best)
    record_event(
        paths.artifact_root, args.run_name, "fusion_evaluate",
        {"best": best, "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest.resolve())},
    )
    print(ranking.head(15).to_string(index=False))
    print(f"Saved: {output_dir / 'best_fusion.json'}")


if __name__ == "__main__":
    main()
