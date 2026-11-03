#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
MANIFEST="${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv"; cd "${PROJECT_ROOT}"; mkdir -p "${ARTIFACT_ROOT}/logs"
folds=("$@"); [[ ${#folds[@]} -gt 0 ]] || folds=(0 1 2 3)
for index in "${!folds[@]}"; do
  fold="${folds[$index]}"; gpu=$((index % 4)); session="m1_cache_f${fold}"
  det_run="m1_d3_fold${fold}"; sem_run="m1_s2_fold${fold}"
  command="source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; set -o pipefail; ${PYTHON} -m filament.build_oof_bank --manifest ${MANIFEST} --fold ${fold} --run-name ${det_run} --checkpoint ${ARTIFACT_ROOT}/runs/${det_run}/weights/best.pt --bank-name m1_oof_yolov8l --imgsz 2048 --base-conf 0.03 --nms-iou 0.75 --max-det 200 --device ${gpu}; ${PYTHON} -m filament.build_semantic_oof --manifest ${MANIFEST} --fold ${fold} --run-name ${sem_run} --checkpoint ${ARTIFACT_ROOT}/runs/${sem_run}/best.pt --cache-name m1_oof_s2 --device ${gpu} 2>&1 | tee ${ARTIFACT_ROOT}/logs/m1_cache_fold${fold}.log"
  tmux has-session -t "${session}" 2>/dev/null && { echo "Exists: ${session}"; continue; }
  tmux new-session -d -s "${session}" "bash -lc '${command}'"; echo "Started ${session} on GPU ${gpu}"
done
tmux ls
