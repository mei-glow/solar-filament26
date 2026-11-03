#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
cd "${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/logs"
[[ -x "${PYTHON}" ]] || { echo "Missing finetune-sd Python: ${PYTHON}" >&2; exit 1; }

launch() {
  local session="$1" config="$2" log="$3"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "bash -lc 'set -o pipefail; source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${PYTHON} -m filament.train_semantic --config ${config} 2>&1 | tee ${log}'"
  echo "Started ${session}"
}

launch p3_s4 configs/p3_s4_semantic_frangi_full.yaml "${ARTIFACT_ROOT}/logs/p3_s4_gpu0.log"
launch p3_s5 configs/p3_s5_semantic_iterative_residual.yaml "${ARTIFACT_ROOT}/logs/p3_s5_gpu1.log"
tmux ls
