#!/usr/bin/env bash
set -euo pipefail
FOLD="${1:?Usage: bash scripts/infer_p0_p1_p2_single.sh FOLD}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"; export PYTHONPATH="${PROJECT_ROOT}/src"; export PYTHONUNBUFFERED=1
"${PYTHON}" -u -m filament.infer_gated_single \
  --fold "${FOLD}" \
  --selector-run m5_p0p1p2_m2 \
  --method m2 \
  --run-name "m5_p0p1p2_single_fold${FOLD}"
