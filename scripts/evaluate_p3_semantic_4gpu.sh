#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
MANIFEST="${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

evaluate_one() {
  local gpu="$1" run="$2" batch="$3"
  python -m filament.evaluate_semantic \
    --run-name "${run}" --manifest "${MANIFEST}" --device "${gpu}" --batch "${batch}" \
    >"${ARTIFACT_ROOT}/logs/${run}_eval.log" 2>&1
}

evaluate_one 0 p3_s0_semantic_unet_r34 4 & PID0=$!
evaluate_one 1 p3_s1_semantic_unetpp_b3 2 & PID1=$!
evaluate_one 2 p3_s2_semantic_skeleton_recall 2 & PID2=$!
evaluate_one 3 p3_s3_semantic_cldice_boundary 1 & PID3=$!
echo "E0=${PID0} E1=${PID1} E2=${PID2} E3=${PID3}"
wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "All full-image semantic evaluations completed."
