from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm
from ultralytics import YOLO

from .config import add_path_args, discover_data, paths_from_args
from .experiment import sha256_file, write_json
from .oof_bank import OOFProposalBank, proposal_record
from .postprocess import rle_from_json
from .semantic_inference import SemanticPredictionCache, infer_semantic_image, load_semantic_checkpoint
from .sira_inference import load_sira, refine_candidates


FEATURES = [
    "detector_score", "mean_probability", "q90_probability", "seed_agreement",
    "semantic_support", "boundary_uncertainty", "elongation", "radial_position",
    "area", "components",
]


def _test_images(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in {".jpeg", ".jpg", ".png"})


def _quality_features(image: np.ndarray, item: dict, semantic: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    box_x0, box_y0, box_x1, box_y1 = map(float, item["box"])
    box_width, box_height = max(box_x1 - box_x0, 1.0), max(box_y1 - box_y0, 1.0)
    center_x, center_y = (box_x0 + box_x1) / 2.0, (box_y0 + box_y1) / 2.0
    roi_probability = item["probability"]
    return {
        "detector_score": float(item["score"]),
        "mean_probability": float(item["mean_probability"]),
        "q90_probability": float(item["q90_probability"]),
        "seed_agreement": float(item["seed_agreement"]),
        "semantic_support": float(semantic[mask].mean()) if mask.any() else 0.0,
        "boundary_uncertainty": float((1.0 - 2.0 * np.abs(roi_probability - 0.5)).clip(0, 1).mean()),
        "elongation": float(max(box_width, box_height) / min(box_width, box_height)),
        "radial_position": float(
            np.hypot(center_x - image.shape[1] / 2, center_y - image.shape[0] / 2)
            / (min(image.shape) / 2)
        ),
        "area": float(mask.sum()),
        "components": float(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fold of M2/M3 SIRA test inference into a resumable bank")
    add_path_args(parser)
    parser.add_argument("--method", choices=("m2", "m3"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-checkpoint", type=Path, default=None)
    parser.add_argument("--semantic-checkpoint", type=Path, default=None)
    parser.add_argument("--sira-checkpoint", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--base-conf", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.75)
    parser.add_argument("--max-det", type=int, default=200)
    parser.add_argument("--semantic-overlap", type=float, default=0.5)
    parser.add_argument("--semantic-batch", type=int, default=4)
    parser.add_argument("--sira-batch", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = paths_from_args(args)
    _, _, test_image_dir = discover_data(paths)
    fold = args.fold
    detector_checkpoint = (args.detector_checkpoint or paths.run_dir(f"m1_d3_fold{fold}") / "weights" / "best.pt").resolve()
    semantic_checkpoint = (args.semantic_checkpoint or paths.run_dir(f"m1_s2_fold{fold}") / "best.pt").resolve()
    sira_checkpoint = (args.sira_checkpoint or paths.run_dir(f"{args.method}_sira_fold{fold}") / "best.pt").resolve()
    for checkpoint in (detector_checkpoint, semantic_checkpoint, sira_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    images = _test_images(test_image_dir)
    if not images:
        raise RuntimeError(f"No test images under {test_image_dir}")
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    semantic_model, semantic_config = load_semantic_checkpoint(semantic_checkpoint, device)
    sira_model, sira_config = load_sira(sira_checkpoint, device)
    semantic_cache = SemanticPredictionCache(
        paths.artifact_root / "predictions" / "semantic_test" / f"m1_s2_fold{fold}",
        semantic_checkpoint,
        {
            "tile_size": int(semantic_config["tile_size"]),
            "overlap": args.semantic_overlap,
            "feature_mode": semantic_config["feature_mode"],
            "tta": False,
        },
    )
    proposal_bank = OOFProposalBank(paths.artifact_root / "predictions" / "test_proposals" / "m1_d3_5fold" / f"fold{fold}")
    refined_bank = OOFProposalBank(paths.artifact_root / "predictions" / "sira_test" / args.method / f"fold{fold}")
    detector_model = None

    for image_path in tqdm(images, desc=f"{args.method.upper()} test fold {fold}", unit="image"):
        if refined_bank.path_for(image_path.name).is_file() and not args.overwrite:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)

        if semantic_cache.has(image_path.name):
            semantic = semantic_cache.load(image_path.name)
        else:
            semantic = infer_semantic_image(
                semantic_model, image, device, int(semantic_config["tile_size"]),
                args.semantic_overlap, semantic_config["feature_mode"], args.semantic_batch, False,
            )
            semantic_cache.save(image_path.name, semantic)

        if proposal_bank.path_for(image_path.name).is_file() and not args.overwrite:
            candidates = proposal_bank.load(image_path.name, decode_masks=False)["instances"]
        else:
            if detector_model is None:
                detector_model = YOLO(str(detector_checkpoint))
            result = detector_model.predict(
                str(image_path), imgsz=args.imgsz, conf=args.base_conf, iou=args.nms_iou,
                max_det=args.max_det, retina_masks=True, device=args.device, verbose=False,
            )[0]
            candidates = []
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy().astype(bool)
                scores = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                candidates = [
                    proposal_record(mask, score, box, f"m1_d3_fold{fold}")
                    for mask, score, box in zip(masks, scores, boxes)
                ]
            proposal_bank.save(image_path.name, fold, candidates, {"checkpoint": str(detector_checkpoint)})

        refined = refine_candidates(sira_model, sira_config, image, candidates, semantic[0], device, args.sira_batch)
        saved = []
        for item in refined:
            x0, y0, x1, y1 = item["bounds"]
            mask = np.zeros(image.shape, dtype=bool)
            mask[y0:y1, x0:x1] = item["probability"] >= 0.5
            record = proposal_record(mask, item["score"], np.asarray(item["box"]), item["source"])
            record["features"] = _quality_features(image, item, semantic[0], mask)
            saved.append(record)
        refined_bank.save(
            image_path.name, fold, saved,
            {"method": args.method, "sira_checkpoint": str(sira_checkpoint), "proposal_count": len(candidates)},
        )

    write_json(refined_bank.root / "metadata.json", {
        "method": args.method,
        "fold": fold,
        "images": len(images),
        "detector_checkpoint": str(detector_checkpoint),
        "detector_sha256": sha256_file(detector_checkpoint),
        "semantic_checkpoint": str(semantic_checkpoint),
        "semantic_sha256": sha256_file(semantic_checkpoint),
        "sira_checkpoint": str(sira_checkpoint),
        "sira_sha256": sha256_file(sira_checkpoint),
        "mask_threshold": 0.5,
    })
    print(f"Saved {args.method.upper()} fold {fold} test bank: {refined_bank.root}")


if __name__ == "__main__":
    main()
