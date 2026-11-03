#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
cd "${PROJECT_ROOT}"; mkdir -p "${ARTIFACT_ROOT}/logs"
folds=("$@")
[[ ${#folds[@]} -gt 0 ]] || folds=(0 1 2 3)
for index in "${!folds[@]}"; do
  fold="${folds[$index]}"; gpu=$((index % 4)); session="m1_det_f${fold}"; run="m1_d3_fold${fold}"
  data="${ARTIFACT_ROOT}/datasets/m1_fold${fold}_medoid_s3020/dataset.yaml"
  tmux has-session -t "${session}" 2>/dev/null && { echo "Exists: ${session}"; continue; }
  command="source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${PYTHON} -m filament.train --run-name ${run} --model yolov8l-seg.pt --data-yaml ${data} --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device ${gpu} --mosaic 0 --mixup 0 --translate 0 --scale 0.1 --fliplr 0.5 --flipud 0.5 --close-mosaic 0 2>&1 | tee ${ARTIFACT_ROOT}/logs/${run}.log"
  tmux new-session -d -s "${session}" "bash -lc '${command}'"; echo "Started ${session} on GPU ${gpu}"
done
tmux ls
