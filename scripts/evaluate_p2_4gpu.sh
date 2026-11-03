#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

eval_one() {
  local gpu="$1" run="$2" manifest="$3" size="$4"
  python -m filament.evaluate --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --run-name "${run}" --manifest "${manifest}" --imgsz "${size}" --device "${gpu}" \
    >"${ARTIFACT_ROOT}/logs/${run}_eval.log" 2>&1
  python -m filament.analysis.error_taxonomy --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --run-name "${run}" --manifest "${manifest}" --imgsz "${size}" \
    >>"${ARTIFACT_ROOT}/logs/${run}_eval.log" 2>&1
  python -m filament.analysis.visualize_errors --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --run-name "${run}" --manifest "${manifest}" --imgsz "${size}" \
    >>"${ARTIFACT_ROOT}/logs/${run}_eval.log" 2>&1
  python -m filament.analysis.stratified_metrics --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --run-name "${run}" --manifest "${manifest}" --imgsz "${size}" \
    >>"${ARTIFACT_ROOT}/logs/${run}_eval.log" 2>&1
}

eval_one 0 p2_d2_yolov8l_2048_standard "${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/manifest.csv" 2048 & PID0=$!
eval_one 1 p2_d3_yolov8l_2048_conservative "${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/manifest.csv" 2048 & PID1=$!
eval_one 2 p2_d4_yolov8l_2048_medoid "${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv" 2048 & PID2=$!
eval_one 3 p2_ref_yolo11s_1024_4class "${ARTIFACT_ROOT}/datasets/fold0_maxcomplete_4class_s42/manifest.csv" 1024 & PID3=$!
wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "P2 evaluation and error taxonomy complete."
