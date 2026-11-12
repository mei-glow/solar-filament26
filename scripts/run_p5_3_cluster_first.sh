#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${ARTIFACT_ROOT}/logs"

"${PYTHON}" -u -m filament.train_p5_cluster_selector \
  --run-name p5_3_cluster_first_m2 \
  --p5-run p5_2_graph_selector_m2 \
  --manifest "${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv" \
  --raw-bank-name m1_oof_yolov8l \
  --sira-run-name m2_sira_oof \
  --quality-thresholds 0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
  --duplicate-thresholds 0.30,0.50,0.70,0.85,1.01 \
  --min-area-values 25,100,200,300
