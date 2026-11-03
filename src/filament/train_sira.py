from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import record_event, write_json
from .refiner_model import build_refiner, refiner_loss
from .sira_data import SIRADataset


def _amp(device: torch.device):
    # Compatible with the older torch build on the server.
    return torch.cuda.amp.autocast(enabled=device.type == "cuda")


@torch.no_grad()
def validate(model, loader, device, exclusion_weight: float) -> dict[str, float]:
    model.eval(); intersection = union = denominator = total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device); target = batch["target"].to(device)
        exclusion = batch["exclusion"].to(device); weight = batch["weight"].to(device)
        logits = model(image)
        loss, _ = refiner_loss(logits, target, sample_weight=weight, exclusion=exclusion, exclusion_weight=exclusion_weight)
        prediction = logits.sigmoid() > .5
        intersection += float((prediction & target.bool()).sum()); union += float((prediction | target.bool()).sum())
        denominator += float(prediction.sum() + target.sum()); total_loss += float(loss)
    return {"loss": total_loss / max(len(loader), 1), "iou": intersection / max(union, 1), "dice": 2 * intersection / max(denominator, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="M2/M3 train SIRA from actual OOF detector seeds")
    add_path_args(parser)
    parser.add_argument("--run-name", required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True); parser.add_argument("--semantic-root", type=Path, default=None)
    parser.add_argument("--val-fold", type=int, default=0); parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--architecture", choices=("unet", "unetplusplus"), default="unet")
    parser.add_argument("--encoder", default="resnet34"); parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--crop-size", type=int, default=512); parser.add_argument("--annotation-strategy", default="medoid")
    parser.add_argument("--neighbor-channel", action="store_true"); parser.add_argument("--exclusion-weight", type=float, default=0.0)
    parser.add_argument("--min-proposal-iou", type=float, default=.02); parser.add_argument("--negative-fraction", type=float, default=.35)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=4); parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="0"); parser.add_argument("--seed", type=int, default=3020)
    args = parser.parse_args()
    if args.exclusion_weight > 0 and not args.neighbor_channel:
        raise ValueError("M3 exclusion requires --neighbor-channel")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    paths = paths_from_args(args); annotation, image_dir, _ = discover_data(paths)
    run_dir = paths.run_dir(args.run_name)
    if run_dir.exists(): raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    train_folds = set(range(args.folds)) - {args.val_fold}; val_folds = {args.val_fold}
    common = dict(annotation_path=annotation, image_dir=image_dir, manifest_path=args.manifest.resolve(),
                  bank_root=args.bank_root.resolve(), crop_size=args.crop_size, annotation_strategy=args.annotation_strategy,
                  semantic_root=args.semantic_root.resolve() if args.semantic_root else None,
                  neighbor_channel=args.neighbor_channel, min_proposal_iou=args.min_proposal_iou, seed=args.seed)
    train_set = SIRADataset(split_folds=train_folds, training=True, negative_fraction=args.negative_fraction, **common)
    val_set = SIRADataset(split_folds=val_folds, training=False, negative_fraction=1.0, **common)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    in_channels = 6 if args.neighbor_channel else 5
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = build_refiner(args.architecture, args.encoder, encoder_weights, in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    config = {**vars(args), "annotation_path": str(annotation), "image_dir": str(image_dir), "in_channels": in_channels,
              "train_folds": sorted(train_folds), "val_folds": sorted(val_folds)}
    write_json(run_dir / "config.json", config); history, best = [], -1.0
    record_event(paths.artifact_root, args.run_name, "sira_train_start", {"config": config})
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        progress = tqdm(train_loader, desc=f"SIRA {epoch}/{args.epochs}")
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True); target = batch["target"].to(device, non_blocking=True)
            exclusion = batch["exclusion"].to(device, non_blocking=True); weight = batch["weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp(device):
                logits = model(image)
                loss, components = refiner_loss(logits, target, sample_weight=weight, exclusion=exclusion, exclusion_weight=args.exclusion_weight)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); running += float(loss)
            progress.set_postfix(loss=f"{float(loss):.4f}")
        scheduler.step(); metrics = validate(model, val_loader, device, args.exclusion_weight)
        row = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1), **metrics}; history.append(row); print(json.dumps(row))
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "metrics": metrics}
        torch.save(checkpoint, run_dir / "last.pt")
        if metrics["dice"] > best: best = metrics["dice"]; torch.save(checkpoint, run_dir / "best.pt")
        write_json(run_dir / "history.json", {"epochs": history, "best_candidate_dice": best})
    record_event(paths.artifact_root, args.run_name, "sira_train_complete", {"best_candidate_dice": best})
    print(f"Best candidate checkpoint: {run_dir / 'best.pt'}; run evaluate_sira for official pooled PQ")


if __name__ == "__main__": main()
