#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

infer_one() {
  local gpu="$1" run="$2" size="$3"
  python -m filament.infer --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --run-name "${run}" --imgsz "${size}" --device "${gpu}" \
    >"${ARTIFACT_ROOT}/logs/${run}_infer.log" 2>&1
}

infer_one 0 p2_d2_yolov8l_2048_standard 2048 & PID0=$!
infer_one 1 p2_d3_yolov8l_2048_conservative 2048 & PID1=$!
infer_one 2 p2_d4_yolov8l_2048_medoid 2048 & PID2=$!
infer_one 3 p2_ref_yolo11s_1024_4class 1024 & PID3=$!
wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "Four audited candidate submissions are under ${ARTIFACT_ROOT}/results/<run-name>/"
