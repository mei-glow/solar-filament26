#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
cd "${PROJECT_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "finetune-sd" ]] || { echo "Run: conda activate finetune-sd" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

python -c "from filament.semantic_model import build_semantic_model; build_semantic_model('unet','resnet34','imagenet'); build_semantic_model('unetplusplus','efficientnet-b3','imagenet'); print('Encoder weights cached')"

python -m filament.train_semantic --config configs/p3_s0_semantic_unet_r34.yaml \
  >"${ARTIFACT_ROOT}/logs/p3_s0_gpu0.log" 2>&1 & PID0=$!
python -m filament.train_semantic --config configs/p3_s1_semantic_unetpp_b3.yaml \
  >"${ARTIFACT_ROOT}/logs/p3_s1_gpu1.log" 2>&1 & PID1=$!
python -m filament.train_semantic --config configs/p3_s2_semantic_skeleton_recall.yaml \
  >"${ARTIFACT_ROOT}/logs/p3_s2_gpu2.log" 2>&1 & PID2=$!
python -m filament.train_semantic --config configs/p3_s3_semantic_cldice_boundary.yaml \
  >"${ARTIFACT_ROOT}/logs/p3_s3_gpu3.log" 2>&1 & PID3=$!

echo "S0=${PID0} S1=${PID1} S2=${PID2} S3=${PID3}"
wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "All semantic/topology branches completed."
