from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

if int(np.__version__.split(".")[0]) >= 2:
    raise RuntimeError(
        f"NumPy {np.__version__} is incompatible with this server's PyTorch build. "
        "Run: python -m pip install --force-reinstall numpy==1.26.4 "
        "opencv-python-headless==4.11.0.86"
    )

from ultralytics import YOLO

from .config import add_config_arg, add_path_args, add_run_args, parse_args_with_config, paths_from_args
from .experiment import record_event, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the YOLOv8l segmentation baseline")
    add_path_args(parser)
    add_run_args(parser, checkpoint=False)
    add_config_arg(parser)
    parser.add_argument("--model", default="yolov8l-seg.pt")
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3020)
    parser.add_argument("--device", default=None, help="Examples: 0 or 0,1. Default auto-detects.")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume last.pt or an explicit checkpoint")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.2)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--overlap-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--single-cls", action=argparse.BooleanOptionalAction, default=True)
    args = parse_args_with_config(parser)
    paths = paths_from_args(args)
    data_yaml = args.data_yaml.resolve() if args.data_yaml else paths.yolo_root / "dataset.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Run filament.prepare first: {data_yaml}")

    device = args.device
    if device is None:
        device = "0,1" if torch.cuda.device_count() >= 2 else ("0" if torch.cuda.is_available() else "cpu")
    run_dir = paths.run_dir(args.run_name)
    last_checkpoint = run_dir / "weights" / "last.pt"
    train_config = vars(args).copy()
    train_config.update({"data_yaml": str(data_yaml), "run_dir": str(run_dir)})
    write_json(paths.experiment_results(args.run_name) / "requested_train_config.json", train_config)
    record_event(
        paths.artifact_root,
        args.run_name,
        "train_start",
        {"config": train_config, "data_sha256": sha256_file(data_yaml)},
    )
    if args.resume is not None:
        resume_checkpoint = last_checkpoint if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        model = YOLO(str(resume_checkpoint))
        model.train(resume=True)
    else:
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Choose a new --run-name or use --resume."
            )
        model = YOLO(args.model)
        kwargs = dict(
            data=str(data_yaml), project=str(paths.runs_root), name=args.run_name, exist_ok=False,
            imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, workers=args.workers,
            device=device, seed=args.seed, deterministic=True, single_cls=args.single_cls,
            cache=False, amp=True, patience=args.patience, save=True, save_period=args.save_period,
            plots=True, close_mosaic=args.close_mosaic, optimizer=args.optimizer,
            mosaic=args.mosaic, mixup=args.mixup, translate=args.translate, scale=args.scale,
            degrees=args.degrees, fliplr=args.fliplr, flipud=args.flipud,
            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
            mask_ratio=args.mask_ratio, overlap_mask=args.overlap_mask,
        )
        if args.lr0 is not None:
            kwargs["lr0"] = args.lr0
        model.train(**kwargs)
    best = run_dir / "weights" / "best.pt"
    record_event(
        paths.artifact_root,
        args.run_name,
        "train_complete",
        {"best_checkpoint": str(best), "best_sha256": sha256_file(best) if best.is_file() else None},
    )
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()

