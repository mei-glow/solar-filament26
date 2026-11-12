#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${ARTIFACT_ROOT}/logs"

"${PYTHON}" -u -m filament.diagnose_pipeline_stages \
  --run-name pipeline_stage_audit_m2 \
  --manifest "${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv" \
  --p5-1-run p5_1_oracle_supervision_m2 \
  --p5-2-run p5_2_graph_selector_m2 \
  --p5-3-run p5_3_cluster_first_m2 \
  --raw-bank-name m1_oof_yolov8l \
  --sira-run-name m2_sira_oof \
  --semantic-cache-name m1_oof_s2 \
  --gallery-size 12
