#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"
BASE_CHECKPOINT="${1:-${ARTIFACT_ROOT}/runs/yolov8l_seg_grouped_baseline/weights/best.pt}"

cd "${PROJECT_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "finetune-sd" ]] || { echo "Run: conda activate finetune-sd" >&2; exit 1; }
[[ -f "${BASE_CHECKPOINT}" ]] || { echo "Missing baseline checkpoint: ${BASE_CHECKPOINT}" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

python -m filament.train --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p2_d2_yolov8l_2048_standard --model "${BASE_CHECKPOINT}" \
  --data-yaml "${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/dataset.yaml" \
  --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device 0 --close-mosaic 5 \
  >"${ARTIFACT_ROOT}/logs/p2_d2_gpu0.log" 2>&1 &
PID0=$!

python -m filament.train --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p2_d3_yolov8l_2048_conservative --model "${BASE_CHECKPOINT}" \
  --data-yaml "${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/dataset.yaml" \
  --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device 1 \
  --mosaic 0 --mixup 0 --translate 0 --scale 0.10 --fliplr 0.5 --flipud 0.5 --close-mosaic 0 \
  >"${ARTIFACT_ROOT}/logs/p2_d3_gpu1.log" 2>&1 &
PID1=$!

python -m filament.train --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p2_d4_yolov8l_2048_medoid --model "${BASE_CHECKPOINT}" \
  --data-yaml "${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/dataset.yaml" \
  --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device 2 \
  --mosaic 0 --mixup 0 --translate 0 --scale 0.10 --fliplr 0.5 --flipud 0.5 --close-mosaic 0 \
  >"${ARTIFACT_ROOT}/logs/p2_d4_gpu2.log" 2>&1 &
PID2=$!

python -m filament.train --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p2_ref_yolo11s_1024_4class --model yolo11s-seg.pt \
  --data-yaml "${ARTIFACT_ROOT}/datasets/fold0_maxcomplete_4class_s42/dataset.yaml" \
  --imgsz 1024 --epochs 30 --batch 4 --workers 6 --device 3 --patience 5 \
  --mask-ratio 2 --no-overlap-mask --no-single-cls --mosaic 0.3 --degrees 10 --scale 0.3 \
  --hsv-v 0.3 --fliplr 0 --flipud 0 --close-mosaic 5 \
  >"${ARTIFACT_ROOT}/logs/p2_ref_gpu3.log" 2>&1 &
PID3=$!

echo "GPU0 PID=${PID0}; GPU1 PID=${PID1}; GPU2 PID=${PID2}; GPU3 PID=${PID3}"
echo "Monitor: watch -n 2 nvidia-smi"
echo "Logs: tail -f ${ARTIFACT_ROOT}/logs/p2_d2_gpu0.log"
wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "All P2 runs completed."
