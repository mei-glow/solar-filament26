from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import add_path_args, paths_from_args
from .experiment import record_event, write_json


TUNED_FIELDS = (
    "detector_conf",
    "semantic_threshold",
    "min_area",
    "rescue_threshold",
    "mean_pq",
    "mean_sq",
    "mean_rq",
    "global_pq",
    "tp",
    "fp",
    "fn",
)


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote one cached fusion-grid row into an inference run")
    add_path_args(parser)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--metric", choices=("global_pq", "mean_pq"), default="global_pq")
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    paths = paths_from_args(args)
    source_dir = paths.results_root / args.source_run
    tuning_path = source_dir / "fusion_tuning.csv"
    config_path = source_dir / "best_fusion.json"
    if not tuning_path.is_file():
        raise FileNotFoundError(tuning_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    ranking = pd.read_csv(tuning_path).sort_values(args.metric, ascending=False)
    if ranking.empty:
        raise RuntimeError(f"No fusion configurations in {tuning_path}")
    best_row = ranking.iloc[0]
    config = json.loads(config_path.read_text())
    for field in TUNED_FIELDS:
        if field in best_row:
            config[field] = _json_value(best_row[field])
    for field in ("min_area", "tp", "fp", "fn"):
        if config.get(field) is not None:
            config[field] = int(config[field])
    config.update({
        "selection_metric": args.metric,
        "promoted_from": args.source_run,
        "fusion_tuning": str(tuning_path.resolve()),
    })

    target_dir = paths.experiment_results(args.target_run)
    output_path = target_dir / "best_fusion.json"
    write_json(output_path, config)
    ranking.to_csv(target_dir / f"fusion_tuning_by_{args.metric}.csv", index=False)
    record_event(
        paths.artifact_root,
        args.target_run,
        "fusion_promote",
        {"source_run": args.source_run, "metric": args.metric, "best": config},
    )
    print(f"\n========== {args.source_run}: TOP {args.metric.upper()} ==========")
    print(ranking.head(args.show).to_string(index=False))
    print(f"\nPromoted: {output_path}")
    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
