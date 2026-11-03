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
from .experiment import record_event, sha256_file, write_json
from .semantic_data import SemanticTileDataset
from .semantic_model import build_semantic_model, semantic_loss


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    intersection = union = denominator = skeleton_hit = skeleton_total = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = model(image)[:, 0:1].sigmoid() >= 0.5
        mask = target[:, 0:1] > 0.5
        skeleton = target[:, 1:2] > 0.5
        intersection += float((prediction & mask).sum())
        union += float((prediction | mask).sum())
        denominator += float(prediction.sum() + mask.sum())
        skeleton_hit += float((prediction & skeleton).sum())
        skeleton_total += float(skeleton.sum())
    return {
        "dice": 2.0 * intersection / max(denominator, 1.0),
        "iou": intersection / max(union, 1.0),
        "skeleton_recall": skeleton_hit / max(skeleton_total, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a full-resolution tiled semantic/topology branch")
    add_path_args(parser)
    add_config_arg(parser)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--manifest", "--split-manifest", dest="manifest", type=Path, default=None)
    parser.add_argument("--architecture", choices=("unet", "unetplusplus", "iterative_unetplusplus"), default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--feature-mode", choices=("radial", "frangi", "intensity"), default="radial")
    parser.add_argument("--annotation-strategy", choices=("duplicate", "medoid", "max-completeness", "first", "random-single"), default="medoid")
    parser.add_argument("--val-annotation-strategy", choices=("duplicate", "medoid", "max-completeness", "first", "random-single"), default="duplicate")
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--val-samples", type=int, default=1200)
    parser.add_argument("--positive-fraction", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--centerline-weight", type=float, default=0.0)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--skeleton-recall-weight", type=float, default=0.0)
    parser.add_argument("--cldice-weight", type=float, default=0.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=3020)
    args = parse_args_with_config(parser)
    if not args.run_name or args.manifest is None:
        parser.error("--run-name and --manifest are required (CLI or YAML config)")
    if not 0 <= args.positive_fraction <= 1:
        raise ValueError("--positive-fraction must be in [0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    paths = paths_from_args(args)
    annotation_path, image_dir, _ = discover_data(paths)
    run_dir = paths.run_dir(args.run_name)
    if run_dir.exists():
        raise FileExistsError(f"Run exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config = vars(args).copy()
    config.update({"annotation_path": str(annotation_path), "image_dir": str(image_dir), "manifest": str(args.manifest.resolve())})
    write_json(run_dir / "config.json", config)
    record_event(
        paths.artifact_root, args.run_name, "semantic_train_start",
        {"config": config, "manifest_sha256": sha256_file(args.manifest.resolve())},
    )

    common = dict(
        annotation_path=annotation_path,
        image_dir=image_dir,
        manifest_path=args.manifest.resolve(),
        tile_size=args.tile_size,
        feature_mode=args.feature_mode,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
    )
    train_dataset = SemanticTileDataset(
        split="train", samples=args.train_samples, annotation_strategy=args.annotation_strategy,
        training=True, **common,
    )
    val_dataset = SemanticTileDataset(
        split="val", samples=args.val_samples, annotation_strategy=args.val_annotation_strategy,
        training=False, **common,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    encoder_weights = None if str(args.encoder_weights).lower() == "none" else args.encoder_weights
    model = build_semantic_model(args.architecture, args.encoder, encoder_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # torch.amp.GradScaler was introduced after torch.cuda.amp.GradScaler.
    # The server's finetune-sd environment uses the older, CUDA-specific API.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict] = []
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        progress = tqdm(train_loader, desc=f"Semantic {epoch}/{args.epochs}")
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(image)
                loss, parts = semantic_loss(
                    logits, target, args.centerline_weight, args.boundary_weight,
                    args.skeleton_recall_weight, args.cldice_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        metrics = validate(model, val_loader, device)
        score = metrics["dice"] + 0.1 * metrics["skeleton_recall"]
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"],
            "selection_score": score,
            **metrics,
        }
        history.append(row)
        print(json.dumps(row))
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": metrics}
        torch.save(checkpoint, run_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, run_dir / "best.pt")
        write_json(run_dir / "history.json", {"epochs": history, "best_score": best_score})
        scheduler.step()

    record_event(
        paths.artifact_root, args.run_name, "semantic_train_complete",
        {"best_checkpoint": str(run_dir / "best.pt"), "best_score": best_score},
    )
    print(f"Best semantic checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
