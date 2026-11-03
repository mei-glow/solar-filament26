#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?Usage: bash scripts/infer_best_fusion.sh <fusion-run-name> [gpu]}"
GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${ZIP_PATH:-${PROJECT_ROOT}/filament-segmentation-2026.zip}"
PYTHON="${PYTHON:-python}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m filament.infer_fusion \
  --zip-path "${ZIP_PATH}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --device "${GPU}" \
  2>&1 | tee "${ARTIFACT_ROOT}/logs/${RUN_NAME}_infer.log"
