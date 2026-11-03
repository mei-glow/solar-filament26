#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"

cd "${PROJECT_ROOT}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "finetune-sd" ]]; then
  echo "ERROR: run 'conda activate finetune-sd' first" >&2
  exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME="fold0_duplicate_s3020"
DATA_YAML="${ARTIFACT_ROOT}/datasets/${DATASET_NAME}/dataset.yaml"
MANIFEST="${ARTIFACT_ROOT}/datasets/${DATASET_NAME}/manifest.csv"
RUN_NAME="yolov8l_seg_grouped_baseline_v2"

python -m filament.prepare --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" --dataset-name "${DATASET_NAME}"
python -m filament.train --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" --run-name "${RUN_NAME}" --data-yaml "${DATA_YAML}" --imgsz 1536 --epochs 60 --batch 2
python -m filament.evaluate --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" --run-name "${RUN_NAME}" --manifest "${MANIFEST}" --imgsz 1536
python -m filament.infer --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" --run-name "${RUN_NAME}" --imgsz 1536

