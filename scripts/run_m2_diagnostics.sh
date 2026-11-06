#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
mkdir -p "${ARTIFACT_ROOT}/logs"
"${PYTHON}" -u -m filament.diagnose_sira \
  --method m2 \
  --manifest "${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv" \
  --quality-run m4_quality_m2 \
  --raw-bank-name m1_oof_yolov8l \
  --sira-run-name m2_sira_oof \
  --count 16
