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
  local session="$1" gpu="$2" run="$3" batch="$4"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "bash -lc 'set -o pipefail; source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${PYTHON} -m filament.infer_fusion --artifact-root ${ARTIFACT_ROOT} --run-name ${run} --device ${gpu} --semantic-batch ${batch} 2>&1 | tee ${ARTIFACT_ROOT}/logs/${run}_infer.log'"
  echo "Started ${session} on GPU ${gpu}"
}

launch p4_s0_infer 0 p4_fusion_s0 4
launch p4_s2_infer 2 p4_fusion_s2 2
launch p4_s3_infer 3 p4_fusion_s3 1
tmux ls
