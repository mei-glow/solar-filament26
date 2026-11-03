#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
ZIP_PATH="${ZIP_PATH:-${PROJECT_ROOT}/filament-segmentation-2026.zip}"
MANIFEST="${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv"
D3_RUN="p2_d3_yolov8l_2048_conservative"
D4_RUN="p2_d4_yolov8l_2048_medoid"
D3="${ARTIFACT_ROOT}/runs/${D3_RUN}/weights/best.pt"
D4="${ARTIFACT_ROOT}/runs/${D4_RUN}/weights/best.pt"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

# Populate low-confidence detector caches on the exact validation split.
"${PYTHON}" -m filament.evaluate --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name "${D3_RUN}" --checkpoint "${D3}" --manifest "${MANIFEST}" \
  --imgsz 2048 --base-conf 0.05 --nms-iou 0.70 --max-det 100 --device 0 \
  --conf-values 0.30 --area-values 200 \
  >"${ARTIFACT_ROOT}/logs/${D3_RUN}_cache.log" 2>&1 & PID_D3=$!

"${PYTHON}" -m filament.evaluate --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name "${D4_RUN}" --checkpoint "${D4}" --manifest "${MANIFEST}" \
  --imgsz 2048 --base-conf 0.05 --nms-iou 0.70 --max-det 100 --device 1 \
  --conf-values 0.30 --area-values 200 \
  >"${ARTIFACT_ROOT}/logs/${D4_RUN}_cache.log" 2>&1 & PID_D4=$!

wait "${PID_D3}" "${PID_D4}"

# Lock the exact submitted S2 fusion parameters instead of re-tuning them.
"${PYTHON}" -m filament.evaluate_fusion \
  --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p4_fusion_s2 --manifest "${MANIFEST}" \
  --semantic-run-name p3_s2_semantic_skeleton_recall \
  --detector-run-names "${D3_RUN},${D4_RUN}" \
  --detector-checkpoints "${D3},${D4}" --device 2 --semantic-batch 2 \
  --detector-imgsz 2048 --base-conf 0.05 --nms-iou 0.70 --max-det 100 \
  --detector-conf-values 0.30 --semantic-threshold-values 0.55 \
  --min-area-values 200 --rescue-threshold-values none \
  --expansion 0.20 --cluster-iou 0.45 --centerline-boost 0.10 \
  --selection-metric global_pq \
  2>&1 | tee "${ARTIFACT_ROOT}/logs/p4_fusion_s2_evaluate.log"

cat "${ARTIFACT_ROOT}/results/p4_fusion_s2/best_fusion.json"
