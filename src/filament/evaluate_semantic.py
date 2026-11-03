from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import record_event, write_json
from .semantic_data import polygons_to_union
from .semantic_inference import SemanticPredictionCache, infer_semantic_image, load_semantic_checkpoint


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    dice = 2.0 * intersection / max(gt.sum() + pred.sum(), 1)
    iou = intersection / max(union, 1)
    gt_skeleton = skeletonize(gt)
    pred_skeleton = skeletonize(pred)
    topology_precision = np.logical_and(pred_skeleton, gt).sum() / max(pred_skeleton.sum(), 1)
    topology_recall = np.logical_and(gt_skeleton, pred).sum() / max(gt_skeleton.sum(), 1)
    cldice = 2 * topology_precision * topology_recall / max(topology_precision + topology_recall, 1e-8)
    return {"dice": dice, "iou": iou, "cldice": cldice, "skeleton_recall": topology_recall}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-image semantic/topology predictions")
    add_path_args(parser)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--threshold-values", default="0.30,0.40,0.50,0.60,0.70")
    args = parser.parse_args()
    paths = paths_from_args(args)
    annotation_path, train_image_dir, _ = discover_data(paths)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else paths.run_dir(args.run_name) / "best.pt"
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model, config = load_semantic_checkpoint(checkpoint, device)
    tile_size = args.tile_size or int(config["tile_size"])
    feature_mode = config["feature_mode"]
    cache_key = f"{args.run_name}_t{tile_size}_o{int(round(args.overlap * 100))}_tta{int(args.tta)}"
    cache = SemanticPredictionCache(
        paths.artifact_root / "predictions" / "semantic" / cache_key / "val",
        checkpoint,
        {"tile_size": tile_size, "overlap": args.overlap, "feature_mode": feature_mode, "tta": args.tta},
    )
    manifest = pd.read_csv(args.manifest.resolve()).query("split == 'val'")
    with annotation_path.open() as handle:
        coco = json.load(handle)
    images = {str(image["id"]): image for image in coco["images"]}
    annotations: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)
    thresholds = [float(value) for value in args.threshold_values.split(",")]
    totals = {threshold: defaultdict(float) for threshold in thresholds}
    rows = []
    for file_name, records in tqdm(manifest.groupby("file_name", sort=True), desc="Full semantic evaluation"):
        if cache.has(file_name):
            output = cache.load(file_name)
        else:
            image = cv2.imread(str(train_image_dir / file_name), cv2.IMREAD_GRAYSCALE)
            output = infer_semantic_image(
                model, image, device, tile_size, args.overlap, feature_mode,
                batch_size=args.batch, tta=args.tta,
            )
            cache.save(file_name, output)
        probability = output[0]
        for record in records.to_dict("records"):
            image_id = str(record["image_id"])
            info = images[image_id]
            gt = polygons_to_union((info["height"], info["width"]), annotations[image_id]).astype(bool)
            for threshold in thresholds:
                values = _metrics(gt, probability >= threshold)
                rows.append({"file_name": file_name, "image_id": image_id, "threshold": threshold, **values})
                for key, value in values.items():
                    totals[threshold][key] += value
                totals[threshold]["count"] += 1
    summary = []
    for threshold, values in totals.items():
        count = values["count"]
        summary.append({"threshold": threshold, **{key: values[key] / count for key in ("dice", "iou", "cldice", "skeleton_recall")}})
    ranking = pd.DataFrame(summary).sort_values("dice", ascending=False)
    output_dir = paths.experiment_results(args.run_name)
    pd.DataFrame(rows).to_csv(output_dir / "semantic_metrics_by_annotation.csv", index=False)
    ranking.to_csv(output_dir / "semantic_tuning.csv", index=False)
    best = ranking.iloc[0].to_dict()
    write_json(output_dir / "best_semantic.json", best)
    record_event(paths.artifact_root, args.run_name, "semantic_evaluate", {"best": best})
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
