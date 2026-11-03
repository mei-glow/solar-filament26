#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"; export PYTHONPATH="${PROJECT_ROOT}/src"
"${PYTHON}" -m filament.proposal_oracle \
  --manifest "${ARTIFACT_ROOT}/results/split_manifest.csv" \
  --run-name p1_proposal_oracle_d3_d4_legacy_val \
  --detector-run-names p2_d3_yolov8l_2048_conservative,p2_d4_yolov8l_2048_medoid \
  --detector-checkpoints "${ARTIFACT_ROOT}/runs/p2_d3_yolov8l_2048_conservative/weights/best.pt,${ARTIFACT_ROOT}/runs/p2_d4_yolov8l_2048_medoid/weights/best.pt" \
  --imgsz 2048 --base-conf 0.05 --nms-iou 0.70 --max-det 100 --device 0
