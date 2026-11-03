#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
cd "${PROJECT_ROOT}"; mkdir -p "${ARTIFACT_ROOT}/logs"
folds=("$@")
[[ ${#folds[@]} -gt 0 ]] || folds=(0 1 2 3)
for index in "${!folds[@]}"; do
  fold="${folds[$index]}"; gpu=$((index % 4)); session="m1_sem_f${fold}"; run="m1_s2_fold${fold}"
  manifest="${ARTIFACT_ROOT}/datasets/m1_fold${fold}_medoid_s3020/manifest.csv"
  tmux has-session -t "${session}" 2>/dev/null && { echo "Exists: ${session}"; continue; }
  command="source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${PYTHON} -m filament.train_semantic --config configs/p3_s2_semantic_skeleton_recall.yaml --run-name ${run} --manifest ${manifest} --device ${gpu} 2>&1 | tee ${ARTIFACT_ROOT}/logs/${run}.log"
  tmux new-session -d -s "${session}" "bash -lc '${command}'"; echo "Started ${session} on GPU ${gpu}"
done
tmux ls
