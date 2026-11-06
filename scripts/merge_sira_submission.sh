#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?Usage: bash scripts/merge_sira_submission.sh m2|m3 [min-fold-support]}"
SUPPORT="${2:-2}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"; export PYTHONPATH="${PROJECT_ROOT}/src"
"${PYTHON}" -u -m filament.merge_sira_submission \
  --method "${METHOD}" \
  --min-fold-support "${SUPPORT}" \
  --cluster-iou 0.45 \
  --duplicate-iou 0.65 \
  --run-name "m4_${METHOD}_ensemble_s${SUPPORT}"
