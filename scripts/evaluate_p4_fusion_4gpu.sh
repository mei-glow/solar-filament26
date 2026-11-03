#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"
MANIFEST="${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv"
D3="${ARTIFACT_ROOT}/runs/p2_d3_yolov8l_2048_conservative/weights/best.pt"
D4="${ARTIFACT_ROOT}/runs/p2_d4_yolov8l_2048_medoid/weights/best.pt"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

evaluate_one() {
  local gpu="$1" semantic_run="$2" fusion_run="$3" batch="$4"
  python -m filament.evaluate_fusion \
    --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
    --run-name "${fusion_run}" --manifest "${MANIFEST}" \
    --semantic-run-name "${semantic_run}" \
    --detector-run-names p2_d3_yolov8l_2048_conservative,p2_d4_yolov8l_2048_medoid \
    --detector-checkpoints "${D3},${D4}" --device "${gpu}" --semantic-batch "${batch}" \
    --detector-conf-values 0.20,0.25,0.30 \
    --semantic-threshold-values 0.35,0.45,0.55 \
    --min-area-values 100,200 \
    --rescue-threshold-values none \
    >"${ARTIFACT_ROOT}/logs/${fusion_run}.log" 2>&1
}

evaluate_one 0 p3_s0_semantic_unet_r34 p4_fusion_s0 4 & PID0=$!
evaluate_one 1 p3_s1_semantic_unetpp_b3 p4_fusion_s1 2 & PID1=$!
evaluate_one 2 p3_s2_semantic_skeleton_recall p4_fusion_s2 2 & PID2=$!
evaluate_one 3 p3_s3_semantic_cldice_boundary p4_fusion_s3 1 & PID3=$!
echo "F0=${PID0} F1=${PID1} F2=${PID2} F3=${PID3}"
wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "All P4 fusion evaluations completed."
