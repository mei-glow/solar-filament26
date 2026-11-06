#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?Usage: bash scripts/run_m4_calibration.sh m2|m3}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"; export PYTHONPATH="${PROJECT_ROOT}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${ARTIFACT_ROOT}/logs"
echo "[M4] method=${METHOD}"
echo "[M4] project_root=${PROJECT_ROOT}"
echo "[M4] artifact_root=${ARTIFACT_ROOT}"
echo "[M4] python=${PYTHON}"
echo "[M4] started=$(date --iso-8601=seconds)"
"${PYTHON}" -m filament.calibrate_quality \
  --run-name "m4_quality_${METHOD}" --sira-run-name "${METHOD}_sira_oof" \
  --manifest "${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv" \
  --folds 5 --model logistic \
  --threshold-values 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80 \
  --min-area-values 100,200,300
echo "[M4] finished=$(date --iso-8601=seconds)"
