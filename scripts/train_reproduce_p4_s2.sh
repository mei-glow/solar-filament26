#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${ZIP_PATH:-${PROJECT_ROOT}/filament-segmentation-2026.zip}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

BASE_RUN="yolov8l_seg_grouped_baseline"
BASE_CHECKPOINT="${ARTIFACT_ROOT}/runs/${BASE_RUN}/weights/best.pt"

"${PYTHON}" -m filament.train \
  --run-name "${BASE_RUN}" --model yolov8l-seg.pt \
  --data-yaml "${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/dataset.yaml" \
  --imgsz 1536 --epochs 60 --batch 2 --workers 8 --device 0 \
  --seed 3020 --patience 20 --save-period 1 --close-mosaic 10 \
  2>&1 | tee "${ARTIFACT_ROOT}/logs/${BASE_RUN}.log"

[[ -f "${BASE_CHECKPOINT}" ]] || { echo "Missing baseline checkpoint: ${BASE_CHECKPOINT}" >&2; exit 1; }

"${PYTHON}" -m filament.train \
  --run-name p2_d3_yolov8l_2048_conservative --model "${BASE_CHECKPOINT}" \
  --data-yaml "${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/dataset.yaml" \
  --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device 0 \
  --seed 3020 --mosaic 0 --mixup 0 --translate 0 --scale 0.10 \
  --fliplr 0.5 --flipud 0.5 --close-mosaic 0 \
  >"${ARTIFACT_ROOT}/logs/p2_d3_yolov8l_2048_conservative.log" 2>&1 & PID_D3=$!

"${PYTHON}" -m filament.train \
  --run-name p2_d4_yolov8l_2048_medoid --model "${BASE_CHECKPOINT}" \
  --data-yaml "${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/dataset.yaml" \
  --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device 1 \
  --seed 3020 --mosaic 0 --mixup 0 --translate 0 --scale 0.10 \
  --fliplr 0.5 --flipud 0.5 --close-mosaic 0 \
  >"${ARTIFACT_ROOT}/logs/p2_d4_yolov8l_2048_medoid.log" 2>&1 & PID_D4=$!

"${PYTHON}" -m filament.train_semantic \
  --config configs/p3_s2_semantic_skeleton_recall.yaml \
  --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p3_s2_semantic_skeleton_recall \
  --manifest "${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv" \
  --device 2 \
  >"${ARTIFACT_ROOT}/logs/p3_s2_semantic_skeleton_recall.log" 2>&1 & PID_S2=$!

echo "D3 PID=${PID_D3}; D4 PID=${PID_D4}; S2 PID=${PID_S2}"
wait "${PID_D3}" "${PID_D4}" "${PID_S2}"
echo "P4-S2 component training complete."
