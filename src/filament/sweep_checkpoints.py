from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import add_path_args, paths_from_args


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PQ-sweep saved YOLO checkpoints across GPUs")
    add_path_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix", default="d1_checkpoint")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    paths = paths_from_args(args)
    checkpoints = sorted(args.checkpoint_dir.resolve().glob("epoch*.pt"))
    best = args.checkpoint_dir.resolve() / "best.pt"
    if best.is_file():
        checkpoints.append(best)
    if not checkpoints:
        raise FileNotFoundError(f"No epoch*.pt or best.pt in {args.checkpoint_dir}")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    jobs = []
    for index, checkpoint in enumerate(checkpoints):
        run_name = f"{args.prefix}_{checkpoint.stem}"
        command = [
            sys.executable, "-m", "filament.evaluate",
            "--zip-path", str(paths.zip_path), "--artifact-root", str(paths.artifact_root),
            "--run-name", run_name, "--checkpoint", str(checkpoint),
            "--manifest", str(args.manifest.resolve()), "--imgsz", str(args.imgsz),
            "--device", devices[index % len(devices)], "--quick",
        ]
        jobs.append((run_name, checkpoint, command))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, command) for _, _, command in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    rows = []
    for run_name, checkpoint, _ in jobs:
        result = paths.experiment_results(run_name) / "best_postprocess.json"
        values = json.loads(result.read_text())
        rows.append({"run_name": run_name, "checkpoint": str(checkpoint), **values})
    ranking = pd.DataFrame(rows).sort_values("mean_pq", ascending=False)
    output = paths.results_root / f"{args.prefix}_ranking.csv"
    ranking.to_csv(output, index=False)
    print(ranking.head(args.top_k).to_string(index=False))
    print("\nRun the full grid for these top checkpoints using filament.evaluate without --quick.")
    print(f"Ranking: {output}")


if __name__ == "__main__":
    main()
