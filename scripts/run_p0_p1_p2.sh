#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"; export PYTHONPATH="${PROJECT_ROOT}/src"; export PYTHONUNBUFFERED=1
mkdir -p "${ARTIFACT_ROOT}/logs"
"${PYTHON}" -u -m filament.train_gated_selector \
  --run-name m5_p0p1p2_m2 \
  --manifest "${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv" \
  --raw-bank-name m1_oof_yolov8l \
  --sira-run-name m2_sira_oof \
  --action-modes raw,refined,gated \
  --refine-margins 0.00,0.05 \
  --score-thresholds 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
  --cluster-ious 0.30,0.45 \
  --min-area 300 \
  --duplicate-iou 0.65
