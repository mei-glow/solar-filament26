from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from .config import add_path_args, discover_data, paths_from_args
from .experiment import record_event, sha256_file, write_json
from .fusion import fuse_instances
from .postprocess import mask_to_rle
from .semantic_inference import SemanticPredictionCache, infer_semantic_image, load_semantic_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a dual-branch fused Kaggle submission")
    add_path_args(parser)
    parser.add_argument("--run-name", required=True, help="Fusion experiment name containing best_fusion.json")
    parser.add_argument("--fusion-config", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--semantic-batch", type=int, default=4)
    parser.add_argument("--detector-imgsz", type=int, default=2048)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    paths = paths_from_args(args)
    _, _, test_image_dir = discover_data(paths)
    output_dir = paths.experiment_results(args.run_name)
    config_path = args.fusion_config.resolve() if args.fusion_config else output_dir / "best_fusion.json"
    config = json.loads(config_path.read_text())

    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    semantic_checkpoint = Path(config["semantic_checkpoint"])
    semantic_model, semantic_train_config = load_semantic_checkpoint(semantic_checkpoint, device)
    tile_size = int(config["tile_size"])
    feature_mode = config["feature_mode"]
    semantic_cache = SemanticPredictionCache(
        paths.artifact_root / "predictions" / "semantic" / config["semantic_cache_key"] / "test",
        semantic_checkpoint,
        {"tile_size": tile_size, "overlap": float(config["overlap"]), "feature_mode": feature_mode, "tta": bool(config["tta"])},
    )
    detector_checkpoints = [Path(path) for path in config["detector_checkpoints"]]
    detector_models = [YOLO(str(path)) for path in detector_checkpoints]
    submission_rows, counts = [], []

    for image_path in tqdm(sorted(test_image_dir.glob("*.jpeg")), desc="Dual-branch test inference"):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if semantic_cache.has(image_path.name):
            semantic_output = semantic_cache.load(image_path.name)
            if semantic_output.shape[1:] != image.shape:
                raise RuntimeError(
                    f"Cached semantic shape mismatch for {image_path.name}: "
                    f"{semantic_output.shape[1:]} != {image.shape}"
                )
        else:
            semantic_output = infer_semantic_image(
                semantic_model, image, device, tile_size, float(config["overlap"]), feature_mode,
                batch_size=args.semantic_batch, tta=bool(config["tta"]),
            )
            semantic_cache.save(image_path.name, semantic_output)
        detector_predictions = []
        for model in detector_models:
            result = model.predict(
                str(image_path), imgsz=args.detector_imgsz, conf=args.base_conf,
                iou=args.nms_iou, max_det=args.max_det, retina_masks=True,
                device=args.device, verbose=False,
            )[0]
            if result.masks is None:
                masks = np.empty((0, 1, 1), dtype=bool)
                scores = np.empty(0, dtype=np.float32)
            else:
                masks = result.masks.data.cpu().numpy().astype(bool)
                scores = result.boxes.conf.cpu().numpy()
            detector_predictions.append((masks, scores))
        rescue = config.get("rescue_threshold")
        instances = fuse_instances(
            semantic_output, detector_predictions,
            detector_confidence=float(config["detector_conf"]),
            semantic_threshold=float(config["semantic_threshold"]),
            min_area=int(config["min_area"]),
            expansion=float(config["expansion"]),
            cluster_iou=float(config["cluster_iou"]),
            centerline_boost=float(config["centerline_boost"]),
            rescue_threshold=None if rescue is None else float(rescue),
        )
        for index, (mask, score) in enumerate(instances, start=1):
            submission_rows.append({
                "filament_id": f"{image_path.stem}_{index}",
                "segmentation_rle": mask_to_rle(mask),
            })
        counts.append({"file_name": image_path.name, "instances": len(instances)})

    submission_path = output_dir / f"submission_{args.run_name}.csv"
    pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"]).to_csv(
        submission_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    pd.DataFrame(counts).to_csv(output_dir / "test_instance_counts.csv", index=False)
    metadata = {
        "fusion_config": str(config_path),
        "semantic_checkpoint_sha256": sha256_file(semantic_checkpoint),
        "detector_checkpoint_sha256": [sha256_file(path) for path in detector_checkpoints],
        "submission_sha256": sha256_file(submission_path),
        "rows": len(submission_rows),
        "images": len(counts),
    }
    write_json(submission_path.with_suffix(".metadata.json"), metadata)
    record_event(paths.artifact_root, args.run_name, "fusion_infer", metadata)
    print(f"Submission: {submission_path}")
    print(f"Rows: {len(submission_rows)}; images: {len(counts)}")


if __name__ == "__main__":
    main()
