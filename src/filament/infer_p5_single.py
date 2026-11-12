from __future__ import annotations

import argparse
import csv
import json

import joblib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .config import add_path_args, discover_data, paths_from_args
from .experiment import sha256_file, write_json
from .gated_selection import FEATURES, build_candidate_features
from .infer_sira_test_fold import _test_images
from .oof_bank import OOFProposalBank
from .p5_selector import (
    ACTIONS,
    add_candidate_context,
    build_pair_features,
    decode_candidate_set,
    normalize_action_predictions,
    pair_prefilter,
)
from .postprocess import mask_to_rle, rle_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a P5.2 graph-selector single-fold submission")
    add_path_args(parser)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--selector-run", default="p5_2_graph_selector_m2")
    parser.add_argument("--method", choices=("m2",), default="m2")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    paths = paths_from_args(args)
    _, _, test_dir = discover_data(paths)
    selector_dir = paths.results_root / args.selector_run
    config_path = selector_dir / "best_p5_selector.json"
    model_path = selector_dir / "p5_selector.joblib"
    if not config_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Missing P5.2 selector artifacts under {selector_dir}")
    config = json.loads(config_path.read_text())
    bundle = joblib.load(model_path)
    if tuple(bundle["actions"]) != ACTIONS:
        raise RuntimeError(f"Action mismatch: {bundle['actions']} != {ACTIONS}")
    if list(bundle["pair_features"]) == []:
        raise RuntimeError("P5.2 pair feature list is empty")

    raw_bank = OOFProposalBank(
        paths.artifact_root / "predictions" / "test_proposals" / "m1_d3_5fold" / f"fold{args.fold}"
    )
    refined_bank = OOFProposalBank(
        paths.artifact_root / "predictions" / "sira_test" / args.method / f"fold{args.fold}"
    )
    run_name = args.run_name or f"p5_2_single_m2_fold{args.fold}"
    output = paths.experiment_results(run_name)
    submission_rows, count_rows = [], []

    for image_path in tqdm(_test_images(test_dir), desc=f"P5.2 single fold {args.fold}", unit="image"):
        raw_path, refined_path = raw_bank.path_for(image_path.name), refined_bank.path_for(image_path.name)
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if not refined_path.is_file():
            raise FileNotFoundError(refined_path)
        raw_instances = raw_bank.load(image_path.name, decode_masks=False)["instances"]
        refined_instances = refined_bank.load(image_path.name, decode_masks=False)["instances"]
        if len(raw_instances) != len(refined_instances):
            raise RuntimeError(f"Raw/refined mismatch for {image_path.name}")
        if raw_instances:
            base = build_candidate_features(raw_instances, refined_instances)
            candidates = base.assign(
                file_name=image_path.name,
                candidate_index=np.arange(len(base), dtype=int),
            )
            pairs = build_pair_features(image_path.name, raw_instances, refined_instances, candidates)
            candidates = add_candidate_context(candidates, pairs)
            action_values = np.column_stack([
                model.predict(candidates[bundle["candidate_features"]]) for model in bundle["action_models"]
            ])
            action_probabilities = normalize_action_predictions(action_values)
            for index, action in enumerate(ACTIONS):
                candidates[f"predicted_p_{action}"] = action_probabilities[:, index]
            eligible_pairs = pairs.loc[pair_prefilter(pairs, float(bundle["pair_prefilter"]))].copy()
            if len(eligible_pairs):
                eligible_pairs["predicted_duplicate_probability"] = np.clip(
                    bundle["pair_model"].predict(eligible_pairs[bundle["pair_features"]]), 0.0, 1.0
                )
            else:
                eligible_pairs["predicted_duplicate_probability"] = pd.Series(dtype=float)
            selected = decode_candidate_set(
                candidates,
                eligible_pairs,
                float(config["selection_threshold"]),
                float(config["duplicate_threshold"]),
                int(config["min_area"]),
                str(config["decoder_mode"]),
            )
        else:
            selected = []

        for output_index, item in enumerate(selected, start=1):
            candidate_index = int(item["candidate_index"])
            source = raw_instances if item["action"] == "raw" else refined_instances
            mask = mask_utils.decode(rle_from_json(source[candidate_index]["rle"])).astype(bool)
            submission_rows.append({
                "filament_id": f"{image_path.stem}_{output_index}",
                "segmentation_rle": mask_to_rle(mask),
            })
        count_rows.append({
            "file_name": image_path.name,
            "candidates": len(raw_instances),
            "instances": len(selected),
            "raw": sum(item["action"] == "raw" for item in selected),
            "refined": sum(item["action"] == "refined" for item in selected),
        })

    submission_path = output / f"submission_{run_name}.csv"
    pd.DataFrame(submission_rows, columns=["filament_id", "segmentation_rle"]).to_csv(
        submission_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    pd.DataFrame(count_rows).to_csv(output / "test_instance_counts.csv", index=False)
    write_json(submission_path.with_suffix(".metadata.json"), {
        "fold": args.fold,
        "selector_run": args.selector_run,
        "selector_config": config,
        "selector_sha256": sha256_file(model_path),
        "rows": len(submission_rows),
        "images": len(count_rows),
        "submission_sha256": sha256_file(submission_path),
    })
    print(f"Submission: {submission_path}")
    print(f"Rows: {len(submission_rows)}; images: {len(count_rows)}")


if __name__ == "__main__":
    main()
