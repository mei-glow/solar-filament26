#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
cd "${PROJECT_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "finetune-sd" ]] || { echo "Run: conda activate finetune-sd" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

python -m filament.train_semantic --config configs/p3_s4_semantic_frangi_full.yaml \
  >"${ARTIFACT_ROOT}/logs/p3_s4_gpu0.log" 2>&1 & PID0=$!
python -m filament.train_semantic --config configs/p3_s5_semantic_iterative_residual.yaml \
  >"${ARTIFACT_ROOT}/logs/p3_s5_gpu1.log" 2>&1 & PID1=$!
echo "S4=${PID0} S5=${PID1}"
wait "${PID0}" "${PID1}"
echo "Vessel-specific ablations completed."
