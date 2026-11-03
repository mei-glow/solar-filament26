from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .oof_bank import OOFProposalBank, proposal_record
from .postprocess import mask_to_rle_object, polygon_to_mask, pq_stats_rle, resolve_instances_scored
from .sira_inference import load_sira, refine_candidates


def _values(text, cast): return [cast(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct held-out SIRA instances and rank by official pooled PQ")
    add_path_args(parser); parser.add_argument("--run-name", required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, default=None); parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="0"); parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--mask-threshold-values", default="0.35,0.45,0.50,0.55,0.65")
    parser.add_argument("--conf-values", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35")
    parser.add_argument("--min-area-values", default="100,200,300")
    args = parser.parse_args(); paths = paths_from_args(args); annotation_path, image_dir, _ = discover_data(paths)
    manifest = pd.read_csv(args.manifest.resolve()); held = manifest[manifest["fold"] == args.fold]
    with annotation_path.open() as handle: coco = json.load(handle)
    images = {str(x["id"]): x for x in coco["images"]}; anns = defaultdict(list)
    for item in coco["annotations"]: anns[str(item["image_id"])].append(item)
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model, config = load_sira(args.checkpoint.resolve(), device)
    source_bank = OOFProposalBank(args.bank_root.resolve() / f"fold{args.fold}")
    target_root = paths.artifact_root / "predictions" / "sira_oof" / args.run_name / f"fold{args.fold}"
    target_bank = OOFProposalBank(target_root)
    configurations = [(m, c, a) for m in _values(args.mask_threshold_values, float) for c in _values(args.conf_values, float) for a in _values(args.min_area_values, int)]
    stats = {key: dict(iou_sum=0., pq_sum=0., count=0, tp=0, fp=0, fn=0) for key in configurations}
    feature_rows = []
    for file_name, records in tqdm(held.groupby("file_name", sort=True), desc=f"SIRA OOF fold {args.fold}"):
        image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
        payload = source_bank.load(file_name, decode_masks=False)
        if args.semantic_root:
            sem_path = args.semantic_root.resolve() / f"fold{args.fold}" / f"{Path(file_name).stem}.npz"
            with np.load(sem_path) as values: semantic = values["probability"].astype(np.float32)[0]
        else: semantic = np.zeros(image.shape, np.float32)
        refined = refine_candidates(model, config, image, payload["instances"], semantic, device, args.batch)
        saved = []
        # Quality targets are diagnostic OOF labels only.
        all_gt = []
        for record in records.itertuples(index=False):
            info = images[str(record.image_id)]
            all_gt.extend(polygon_to_mask(x["segmentation"], info["height"], info["width"]) for x in anns[str(record.image_id)])
        gt_rles_all = [mask_to_rle_object(x) for x in all_gt]
        for index, item in enumerate(refined):
            x0, y0, x1, y1 = item["bounds"]
            mask = np.zeros(image.shape, bool); mask[y0:y1, x0:x1] = item["probability"] >= .5
            pred_rle = mask_to_rle_object(mask)
            target_iou = float(mask_utils.iou([pred_rle], gt_rles_all, [0] * len(gt_rles_all)).max()) if gt_rles_all else 0.0
            area = int(mask.sum()); components = int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)
            box_x0, box_y0, box_x1, box_y1 = map(float, item["box"])
            box_width, box_height = max(box_x1 - box_x0, 1.), max(box_y1 - box_y0, 1.)
            center_x, center_y = (box_x0 + box_x1) / 2, (box_y0 + box_y1) / 2
            radial = float(np.hypot(center_x - image.shape[1] / 2, center_y - image.shape[0] / 2) / (min(image.shape) / 2))
            roi_probability = item["probability"]
            uncertainty = float((1.0 - 2.0 * np.abs(roi_probability - .5)).clip(0, 1).mean())
            semantic_support = float(semantic[mask].mean()) if mask.any() else 0.0
            row = {"file_name": file_name, "candidate_index": index, "fold": args.fold, "detector_score": item["score"],
                   "source": item["source"], "mean_probability": item["mean_probability"], "q90_probability": item["q90_probability"],
                   "seed_agreement": item["seed_agreement"], "semantic_support": semantic_support,
                   "boundary_uncertainty": uncertainty, "elongation": max(box_width, box_height) / min(box_width, box_height),
                   "radial_position": radial, "area": area, "components": components, "target_iou": target_iou}
            feature_rows.append(row)
            saved_item = proposal_record(mask, item["score"], np.asarray(item["box"]), item["source"])
            saved_item["features"] = {key: value for key, value in row.items() if key not in ("file_name", "candidate_index", "source")}
            saved.append(saved_item)
        target_bank.save(file_name, args.fold, saved, {"checkpoint": str(args.checkpoint.resolve())})
        mask_thresholds = _values(args.mask_threshold_values, float)
        conf_values = _values(args.conf_values, float)
        area_values = _values(args.min_area_values, int)
        for mask_threshold in mask_thresholds:
            masks = np.zeros((len(refined), *image.shape), bool)
            for index, item in enumerate(refined):
                x0, y0, x1, y1 = item["bounds"]
                masks[index, y0:y1, x0:x1] = item["probability"] >= mask_threshold
            scores = np.asarray([item["score"] for item in refined], np.float32)
            for min_area in area_values:
                selected = resolve_instances_scored(masks, scores, min(conf_values), min_area)
                encoded = [(mask_to_rle_object(mask), score) for mask, score in selected]
                for conf in conf_values:
                    pred_rles = [rle for rle, score in encoded if score >= conf]
                    current = stats[(mask_threshold, conf, min_area)]
                    for record in records.itertuples(index=False):
                        info = images[str(record.image_id)]; gt = [mask_to_rle_object(polygon_to_mask(x["segmentation"], info["height"], info["width"])) for x in anns[str(record.image_id)]]
                        values = pq_stats_rle(gt, pred_rles)
                        for key in ("iou_sum", "tp", "fp", "fn"): current[key] += values[key]
                        current["pq_sum"] += values["pq"]; current["count"] += 1
            del masks
    rows = []
    for (mask_threshold, conf, area), values in stats.items():
        denominator = values["tp"] + .5 * values["fp"] + .5 * values["fn"]
        rows.append({"mask_threshold": mask_threshold, "conf": conf, "min_area": area,
                     "global_pq": values["iou_sum"] / denominator if denominator else 0., "mean_pq": values["pq_sum"] / max(values["count"], 1),
                     "tp": values["tp"], "fp": values["fp"], "fn": values["fn"]})
    output = paths.experiment_results(f"{args.run_name}_fold{args.fold}"); ranking = pd.DataFrame(rows).sort_values("global_pq", ascending=False)
    ranking.to_csv(output / "sira_tuning.csv", index=False); pd.DataFrame(feature_rows).to_csv(output / "quality_features.csv", index=False)
    (output / "best_sira.json").write_text(json.dumps(ranking.iloc[0].to_dict(), indent=2) + "\n")
    print(ranking.head(12).to_string(index=False)); print(f"Refined OOF bank: {target_root}")


if __name__ == "__main__": main()
