#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
MANIFEST="${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv"
cd "${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/logs"
[[ -x "${PYTHON}" ]] || { echo "Missing finetune-sd Python: ${PYTHON}" >&2; exit 1; }

launch() {
  local session="$1" gpu="$2" run="$3" batch="$4"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "bash -lc 'set -o pipefail; source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${PYTHON} -m filament.evaluate_semantic --run-name ${run} --manifest ${MANIFEST} --device ${gpu} --batch ${batch} 2>&1 | tee ${ARTIFACT_ROOT}/logs/${run}_eval.log'"
  echo "Started ${session}"
}

launch p3_s4_eval 0 p3_s4_semantic_frangi_full 1
launch p3_s5_eval 1 p3_s5_semantic_iterative_residual 1
tmux ls
