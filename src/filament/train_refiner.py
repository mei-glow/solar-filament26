from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import add_config_arg, add_path_args, discover_data, parse_args_with_config, paths_from_args
from .experiment import record_event, write_json
from .refiner_data import RefinerDataset
from .refiner_model import build_refiner, refiner_loss


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    intersections = unions = dice_denominator = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = model(image).sigmoid() > 0.5
        intersections += float((prediction * target.bool()).sum())
        unions += float((prediction | target.bool()).sum())
        dice_denominator += float(prediction.sum() + target.sum())
    return {
        "iou": intersections / max(unions, 1.0),
        "dice": 2.0 * intersections / max(dice_denominator, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the seed-conditioned high-resolution mask refiner")
    add_path_args(parser)
    add_config_arg(parser)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--manifest", "--split-manifest", dest="manifest", type=Path, default=None)
    parser.add_argument("--architecture", choices=("unet", "unetplusplus"), default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--annotation-strategy", choices=("duplicate", "medoid", "max-completeness", "first", "random-single"), default="medoid")
    parser.add_argument("--val-annotation-strategy", choices=("duplicate", "medoid", "max-completeness", "first", "random-single"), default="duplicate")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--negative-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--skeleton-weight", type=float, default=0.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=3020)
    args = parse_args_with_config(parser)
    if not args.run_name or args.manifest is None:
        parser.error("--run-name and --manifest/--split-manifest are required (CLI or config)")
    if not 0 <= args.negative_fraction < 1:
        raise ValueError("--negative-fraction must be in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    paths = paths_from_args(args)
    annotation_path, image_dir, _ = discover_data(paths)
    run_dir = paths.run_dir(args.run_name)
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config = vars(args).copy()
    config.update({"annotation_path": str(annotation_path), "image_dir": str(image_dir)})
    write_json(run_dir / "config.json", config)
    record_event(paths.artifact_root, args.run_name, "refiner_train_start", {"config": config})

    train_dataset = RefinerDataset(
        annotation_path, image_dir, args.manifest.resolve(), "train", args.crop_size,
        args.annotation_strategy, args.seed, args.negative_fraction, training=True,
    )
    val_dataset = RefinerDataset(
        annotation_path, image_dir, args.manifest.resolve(), "val", args.crop_size,
        args.val_annotation_strategy, args.seed, args.negative_fraction, training=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_refiner(args.architecture, args.encoder, encoder_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_dice = [], -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Refiner {epoch}/{args.epochs}")
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(image)
                loss, components = refiner_loss(logits, target, args.boundary_weight, args.skeleton_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step()
        metrics = validate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / max(len(train_loader), 1), "lr": scheduler.get_last_lr()[0], **metrics}
        history.append(row)
        print(json.dumps(row))
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": metrics}
        torch.save(checkpoint, run_dir / "last.pt")
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(checkpoint, run_dir / "best.pt")
        write_json(run_dir / "history.json", {"epochs": history, "best_dice": best_dice})

    record_event(
        paths.artifact_root, args.run_name, "refiner_train_complete",
        {"best_checkpoint": str(run_dir / "best.pt"), "best_dice": best_dice},
    )
    print(f"Best refiner: {run_dir / 'best.pt'} (ROI Dice={best_dice:.6f})")


if __name__ == "__main__":
    main()
