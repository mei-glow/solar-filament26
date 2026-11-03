#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?Usage: bash scripts/launch_sira_eval_tmux.sh m2|m3 [fold ...]}"; shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-$(command -v python)}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
MANIFEST="${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv"; BANK="${ARTIFACT_ROOT}/predictions/oof/m1_oof_yolov8l"
SEMANTIC="${ARTIFACT_ROOT}/predictions/semantic_oof/m1_oof_s2"; cd "${PROJECT_ROOT}"; mkdir -p "${ARTIFACT_ROOT}/logs"
folds=("$@"); [[ ${#folds[@]} -gt 0 ]] || folds=(0 1 2 3)
for index in "${!folds[@]}"; do
  fold="${folds[$index]}"; gpu=$((index % 4)); model_run="${METHOD}_sira_fold${fold}"; output_run="${METHOD}_sira_oof"; session="${METHOD}_eval_f${fold}"
  command="source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src; ${PYTHON} -m filament.evaluate_sira --run-name ${output_run} --checkpoint ${ARTIFACT_ROOT}/runs/${model_run}/best.pt --manifest ${MANIFEST} --bank-root ${BANK} --semantic-root ${SEMANTIC} --fold ${fold} --device ${gpu} 2>&1 | tee ${ARTIFACT_ROOT}/logs/${METHOD}_eval_fold${fold}.log"
  tmux has-session -t "${session}" 2>/dev/null && { echo "Exists: ${session}"; continue; }
  tmux new-session -d -s "${session}" "bash -lc '${command}'"; echo "Started ${session} on GPU ${gpu}"
done
tmux ls
