from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .config import add_path_args, discover_data, paths_from_args
from .experiment import sha256_file, write_json
from .oof_bank import OOFProposalBank, proposal_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer exactly one held-out fold and build a leakage-safe OOF proposal bank")
    add_path_args(parser)
    parser.add_argument("--manifest", type=Path, required=True, help="Five-fold physical-image manifest")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-name", required=True, help="Fold-specific detector run name")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-name", default="m1_oof_yolov8l")
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--base-conf", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--max-det", type=int, default=200)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = paths_from_args(args)
    _, image_dir, _ = discover_data(paths)
    manifest_path, checkpoint = args.manifest.resolve(), args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = pd.read_csv(manifest_path)
    required = {"file_name", "fold"}
    if not required.issubset(manifest):
        raise ValueError(f"Manifest lacks {required - set(manifest.columns)}")
    if manifest.groupby("file_name")["fold"].nunique().max() != 1:
        raise RuntimeError("Physical-image leakage in manifest")
    files = sorted(manifest.loc[manifest["fold"] == args.fold, "file_name"].unique())
    if not files:
        raise ValueError(f"Fold {args.fold} has no files")
    root = paths.artifact_root / "predictions" / "oof" / args.bank_name / f"fold{args.fold}"
    bank = OOFProposalBank(root)
    model = YOLO(str(checkpoint))
    for file_name in tqdm(files, desc=f"OOF fold {args.fold}"):
        if bank.path_for(file_name).is_file() and not args.overwrite:
            continue
        result = model.predict(
            str(image_dir / file_name), imgsz=args.imgsz, conf=args.base_conf,
            iou=args.nms_iou, max_det=args.max_det, retina_masks=True,
            device=args.device, verbose=False,
        )[0]
        instances = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)
            scores = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            instances = [proposal_record(mask, score, box, args.run_name) for mask, score, box in zip(masks, scores, boxes)]
        bank.save(file_name, args.fold, instances, {"run_name": args.run_name, "checkpoint": str(checkpoint)})

    write_json(root / "metadata.json", {
        "fold": args.fold, "run_name": args.run_name, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path), "files": len(files),
        "imgsz": args.imgsz, "base_conf": args.base_conf, "nms_iou": args.nms_iou, "max_det": args.max_det,
    })
    print(f"OOF bank: {root} ({len(files)} physical images)")


if __name__ == "__main__":
    main()
