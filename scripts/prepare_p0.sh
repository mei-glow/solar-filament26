#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"

cd "${PROJECT_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "finetune-sd" ]] || { echo "Run: conda activate finetune-sd" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

LEGACY_MANIFEST="${ARTIFACT_ROOT}/results/split_manifest.csv"
[[ -f "${LEGACY_MANIFEST}" ]] || { echo "Missing legacy baseline manifest: ${LEGACY_MANIFEST}" >&2; exit 1; }

python -m filament.prepare --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-name legacy_duplicate_s3020 --annotation-strategy duplicate --input-manifest "${LEGACY_MANIFEST}"
python -m filament.prepare --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-name legacy_medoid_s3020 --annotation-strategy medoid --input-manifest "${LEGACY_MANIFEST}"
python -m filament.prepare --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-name fold0_medoid_s3020 --annotation-strategy medoid --val-fold 0 --seed 3020
python -m filament.prepare --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-name fold0_maxcomplete_4class_s42 --annotation-strategy max-completeness \
  --val-annotation-strategy same --use-categories --public-random-split --seed 42
python -m filament.prepare --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-name fold0_randomsingle_s3020 --annotation-strategy random-single --val-fold 0 --seed 3020

echo "Prepared legacy-safe P2 variants and clean five-fold P3/paper variants."
