#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src"
for fold in 0 1 2 3 4; do
  "${PYTHON}" -m filament.prepare \
    --zip-path "${PROJECT_ROOT}/filament-segmentation-2026.zip" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --folds 5 --val-fold "${fold}" --seed 3020 \
    --annotation-strategy medoid --val-annotation-strategy all \
    --dataset-name "m1_fold${fold}_medoid_s3020" \
    --fold-manifest "${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv"
done
echo "Prepared five grouped folds; master manifest: ${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv"
