#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
cd "${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/logs"
[[ -x "${PYTHON}" ]] || { echo "Missing finetune-sd Python: ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m filament.promote_fusion \
  --artifact-root "${ARTIFACT_ROOT}" \
  --source-run p4_fusion_s4 \
  --target-run p4_fusion_s4_global \
  --metric global_pq

"${PYTHON}" -m filament.promote_fusion \
  --artifact-root "${ARTIFACT_ROOT}" \
  --source-run p4_fusion_s5 \
  --target-run p4_fusion_s5_global \
  --metric global_pq

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

launch p4_s4_global_infer 0 p4_fusion_s4_global 1
launch p4_s5_global_infer 1 p4_fusion_s5_global 1
tmux ls
