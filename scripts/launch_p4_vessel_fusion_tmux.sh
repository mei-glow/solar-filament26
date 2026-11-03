#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
MANIFEST="${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv"
D3="${ARTIFACT_ROOT}/runs/p2_d3_yolov8l_2048_conservative/weights/best.pt"
D4="${ARTIFACT_ROOT}/runs/p2_d4_yolov8l_2048_medoid/weights/best.pt"
cd "${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/logs"
[[ -x "${PYTHON}" ]] || { echo "Missing finetune-sd Python: ${PYTHON}" >&2; exit 1; }

launch() {
  local session="$1" gpu="$2" semantic="$3" fusion="$4" batch="$5"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}"
    return
  fi
  local command="${PYTHON} -m filament.evaluate_fusion --run-name ${fusion} --manifest ${MANIFEST} --semantic-run-name ${semantic} --detector-run-names p2_d3_yolov8l_2048_conservative,p2_d4_yolov8l_2048_medoid --detector-checkpoints ${D3},${D4} --device ${gpu} --semantic-batch ${batch} --detector-conf-values 0.20,0.25,0.30 --semantic-threshold-values 0.35,0.45,0.55 --min-area-values 100,200 --rescue-threshold-values none"
  tmux new-session -d -s "${session}" \
    "bash -lc 'set -o pipefail; source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${command} 2>&1 | tee ${ARTIFACT_ROOT}/logs/${fusion}.log'"
  echo "Started ${session}"
}

launch p4_s4 0 p3_s4_semantic_frangi_full p4_fusion_s4 1
launch p4_s5 1 p3_s5_semantic_iterative_residual p4_fusion_s5 1
tmux ls
