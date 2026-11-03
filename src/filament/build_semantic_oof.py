from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .semantic_inference import SemanticPredictionCache, infer_semantic_image, load_semantic_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic predictions for exactly one held-out physical-image fold")
    add_path_args(parser); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-name", required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-name", default="m1_oof_s2"); parser.add_argument("--device", default="0")
    parser.add_argument("--tile-size", type=int, default=None); parser.add_argument("--overlap", type=float, default=.5)
    parser.add_argument("--batch", type=int, default=4); parser.add_argument("--tta", action="store_true")
    args = parser.parse_args(); paths = paths_from_args(args); _, image_dir, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve()); files = sorted(manifest.loc[manifest["fold"] == args.fold, "file_name"].unique())
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model, config = load_semantic_checkpoint(args.checkpoint.resolve(), device)
    tile_size = args.tile_size or int(config["tile_size"]); feature_mode = config["feature_mode"]
    cache = SemanticPredictionCache(paths.artifact_root / "predictions" / "semantic_oof" / args.cache_name / f"fold{args.fold}",
                                    args.checkpoint.resolve(), {"tile_size": tile_size, "overlap": args.overlap, "feature_mode": feature_mode, "tta": args.tta})
    for file_name in tqdm(files, desc=f"Semantic OOF fold {args.fold}"):
        if cache.has(file_name): continue
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        output = infer_semantic_image(model, image, device, tile_size, args.overlap, feature_mode, args.batch, args.tta)
        cache.save(file_name, output)
    print(f"Semantic OOF cache: {cache.root} ({len(files)} files)")


if __name__ == "__main__": main()
