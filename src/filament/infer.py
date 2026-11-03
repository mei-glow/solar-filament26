from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .config import add_config_arg, add_path_args, add_run_args, discover_data, parse_args_with_config, paths_from_args, resolve_checkpoint
from .experiment import record_event, sha256_file, write_json
from .postprocess import mask_to_rle, resolve_instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test inference and create a Kaggle submission")
    add_path_args(parser)
    add_run_args(parser)
    add_config_arg(parser)
    parser.add_argument("--postprocess", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    args = parse_args_with_config(parser)
    paths = paths_from_args(args)
    _, _, test_image_dir = discover_data(paths)
    checkpoint = resolve_checkpoint(paths, args.run_name, args.checkpoint)
    output_dir = paths.experiment_results(args.run_name)
    tuning_path = args.postprocess.resolve() if args.postprocess else output_dir / "best_postprocess.json"
    if not tuning_path.is_file():
        raise FileNotFoundError(f"Run evaluate first or pass --postprocess: {tuning_path}")
    params = json.loads(tuning_path.read_text())
    confidence, min_area = float(params["conf"]), int(params["min_area"])
    model = YOLO(str(checkpoint))

    submission_rows, count_rows = [], []
    for image_path in tqdm(sorted(test_image_dir.glob("*.jpeg")), desc="Test inference"):
        result = model.predict(
            str(image_path), imgsz=args.imgsz, conf=args.base_conf, iou=args.nms_iou,
            max_det=args.max_det, retina_masks=True, device=args.device, verbose=False,
        )[0]
        if result.masks is None:
            instances = []
        else:
            instances = resolve_instances(
                result.masks.data.cpu().numpy().astype(bool), result.boxes.conf.cpu().numpy(),
                conf=confidence, min_area=min_area,
            )
        for index, mask in enumerate(instances, start=1):
            submission_rows.append(
                {"filament_id": f"{image_path.stem}_{index}", "segmentation_rle": mask_to_rle(mask)}
            )
        count_rows.append({"file_name": image_path.name, "instances": len(instances)})

    submission_path = output_dir / f"submission_{args.run_name}.csv"
    pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"]).to_csv(
        submission_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    pd.DataFrame(count_rows).to_csv(output_dir / "test_instance_counts.csv", index=False)
    metadata = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "postprocess": str(tuning_path),
        "postprocess_parameters": params,
        "imgsz": args.imgsz,
        "base_conf": args.base_conf,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "rows": len(submission_rows),
        "images": len(count_rows),
    }
    metadata["submission_sha256"] = sha256_file(submission_path)
    write_json(submission_path.with_suffix(".metadata.json"), metadata)
    record_event(paths.artifact_root, args.run_name, "infer", metadata)
    print(f"Submission: {submission_path}")
    print(f"Rows: {len(submission_rows)}; images: {len(count_rows)}")


if __name__ == "__main__":
    main()

