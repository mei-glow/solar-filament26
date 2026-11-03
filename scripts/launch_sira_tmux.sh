#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?Usage: bash scripts/launch_sira_tmux.sh m2|m3 [fold ...]}"; shift
[[ "${METHOD}" == "m2" || "${METHOD}" == "m3" ]] || { echo "METHOD must be m2 or m3" >&2; exit 2; }
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
MANIFEST="${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv"; BANK="${ARTIFACT_ROOT}/predictions/oof/m1_oof_yolov8l"
SEMANTIC="${ARTIFACT_ROOT}/predictions/semantic_oof/m1_oof_s2"; cd "${PROJECT_ROOT}"; mkdir -p "${ARTIFACT_ROOT}/logs"
folds=("$@"); [[ ${#folds[@]} -gt 0 ]] || folds=(0 1 2 3)
extra=""; [[ "${METHOD}" == "m3" ]] && extra="--neighbor-channel --exclusion-weight 0.25"
for index in "${!folds[@]}"; do
  fold="${folds[$index]}"; gpu=$((index % 4)); run="${METHOD}_sira_fold${fold}"; session="${METHOD}_f${fold}"
  command="source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${PYTHON} -m filament.train_sira --run-name ${run} --manifest ${MANIFEST} --bank-root ${BANK} --semantic-root ${SEMANTIC} --val-fold ${fold} --crop-size 512 --architecture unet --encoder resnet34 --epochs 25 --batch 4 --workers 6 --device ${gpu} ${extra} 2>&1 | tee ${ARTIFACT_ROOT}/logs/${run}.log"
  tmux has-session -t "${session}" 2>/dev/null && { echo "Exists: ${session}"; continue; }
  tmux new-session -d -s "${session}" "bash -lc '${command}'"; echo "Started ${session} on GPU ${gpu}"
done
tmux ls
