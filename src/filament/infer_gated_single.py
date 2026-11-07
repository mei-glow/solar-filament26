from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import sha256_file, write_json
from .gated_selection import FEATURES, build_candidate_features, choose_candidate_rles, select_set_aware_instances
from .infer_sira_test_fold import _test_images
from .oof_bank import OOFProposalBank
from .postprocess import mask_to_rle
from pycocotools import mask as mask_utils


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-ensemble single-fold P0/P1/P2 submission")
    add_path_args(parser)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--selector-run", default="m5_p0p1p2_m2")
    parser.add_argument("--method", default="m2")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    paths = paths_from_args(args); _, _, test_dir = discover_data(paths)
    selector_dir = paths.results_root / args.selector_run
    config_path = selector_dir / "best_gate_setaware.json"; model_path = selector_dir / "gate_ranker.joblib"
    config = json.loads(config_path.read_text()); bundle = joblib.load(model_path)
    if list(bundle["features"]) != FEATURES: raise RuntimeError("Gate/ranker feature mismatch")
    raw_bank = OOFProposalBank(paths.artifact_root / "predictions" / "test_proposals" / "m1_d3_5fold" / f"fold{args.fold}")
    refined_bank = OOFProposalBank(paths.artifact_root / "predictions" / "sira_test" / args.method / f"fold{args.fold}")
    run_name = args.run_name or f"m5_p0p1p2_single_fold{args.fold}"
    output = paths.experiment_results(run_name); submission_rows, count_rows = [], []

    for image_path in tqdm(_test_images(test_dir), desc=f"P0/P1/P2 single fold {args.fold}", unit="image"):
        raw_path, refined_path = raw_bank.path_for(image_path.name), refined_bank.path_for(image_path.name)
        if not raw_path.is_file(): raise FileNotFoundError(raw_path)
        if not refined_path.is_file(): raise FileNotFoundError(refined_path)
        raw_instances = raw_bank.load(image_path.name, decode_masks=False)["instances"]
        refined_instances = refined_bank.load(image_path.name, decode_masks=False)["instances"]
        if raw_instances:
            features = build_candidate_features(raw_instances, refined_instances)
            predicted_raw = np.clip(bundle["raw_model"].predict(features[FEATURES]), 0, 1)
            predicted_refined = np.clip(bundle["refined_model"].predict(features[FEATURES]), 0, 1)
            chosen, scores, use_refined = choose_candidate_rles(
                raw_instances, refined_instances, predicted_raw, predicted_refined,
                str(config["action_mode"]), float(config["refine_margin"]),
            )
            selected = select_set_aware_instances(
                chosen, scores, float(config["score_threshold"]), int(config["min_area"]),
                float(config["cluster_iou"]), float(config["duplicate_iou"]),
            )
        else:
            use_refined = np.empty(0, dtype=bool); selected = []
        for index, (rle, _) in enumerate(selected, start=1):
            mask = mask_utils.decode(rle).astype(bool)
            submission_rows.append({"filament_id": f"{image_path.stem}_{index}", "segmentation_rle": mask_to_rle(mask)})
        count_rows.append({
            "file_name": image_path.name, "raw_candidates": len(raw_instances),
            "refined_fraction": float(use_refined.mean()) if len(use_refined) else 0.0,
            "instances": len(selected),
        })

    submission_path = output / f"submission_{run_name}.csv"
    pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"]).to_csv(submission_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(count_rows).to_csv(output / "test_instance_counts.csv", index=False)
    write_json(submission_path.with_suffix(".metadata.json"), {
        "fold": args.fold, "selector_run": args.selector_run, "selector_config": config,
        "selector_sha256": sha256_file(model_path), "rows": len(submission_rows),
        "images": len(count_rows), "submission_sha256": sha256_file(submission_path),
    })
    print(f"Submission: {submission_path}")
    print(f"Rows: {len(submission_rows)}; images: {len(count_rows)}")


if __name__ == "__main__":
    main()
