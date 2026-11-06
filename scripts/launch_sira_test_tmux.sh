#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?Usage: bash scripts/launch_sira_test_tmux.sh m2|m3 [fold ...]}"; shift
if [[ "${METHOD}" != "m2" && "${METHOD}" != "m3" ]]; then
  echo "METHOD must be m2 or m3" >&2; exit 2
fi
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
cd "${PROJECT_ROOT}"; mkdir -p "${ARTIFACT_ROOT}/logs"
folds=("$@"); [[ ${#folds[@]} -gt 0 ]] || folds=(0 1 2 3)
for index in "${!folds[@]}"; do
  fold="${folds[$index]}"; gpu=$((index % 4)); session="${METHOD}_test_f${fold}"
  command="source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PROJECT_ROOT=${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; set -o pipefail; ${PYTHON} -u -m filament.infer_sira_test_fold --method ${METHOD} --fold ${fold} --device ${gpu} 2>&1 | tee ${ARTIFACT_ROOT}/logs/${METHOD}_test_fold${fold}.log"
  tmux has-session -t "${session}" 2>/dev/null && { echo "Exists: ${session}"; continue; }
  tmux new-session -d -s "${session}" "bash -lc '${command}'"
  echo "Started ${session} on GPU ${gpu}"
done
tmux ls
