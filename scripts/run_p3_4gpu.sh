#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"
MANIFEST="${ARTIFACT_ROOT}/datasets/fold0_medoid_s3020/manifest.csv"
cd "${PROJECT_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "finetune-sd" ]] || { echo "Run: conda activate finetune-sd" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${ARTIFACT_ROOT}/logs"

# Download/cache encoder weights once before concurrent workers start.
python -c "from filament.refiner_model import build_refiner; build_refiner('unet','resnet34','imagenet'); build_refiner('unetplusplus','efficientnet-b3','imagenet'); print('Encoder weights cached')"

python -m filament.train_refiner --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p3_r0_unet_r34_bce_dice --manifest "${MANIFEST}" --architecture unet --encoder resnet34 \
  --crop-size 512 --batch 6 --epochs 30 --device 0 >"${ARTIFACT_ROOT}/logs/p3_r0_gpu0.log" 2>&1 & PID0=$!

python -m filament.train_refiner --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p3_r1_unetpp_b3_bce_dice --manifest "${MANIFEST}" --architecture unetplusplus --encoder efficientnet-b3 \
  --crop-size 512 --batch 4 --epochs 30 --device 1 >"${ARTIFACT_ROOT}/logs/p3_r1_gpu1.log" 2>&1 & PID1=$!

python -m filament.train_refiner --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p3_r2_unetpp_b3_boundary --manifest "${MANIFEST}" --architecture unetplusplus --encoder efficientnet-b3 \
  --crop-size 512 --batch 4 --epochs 30 --boundary-weight 3.0 --device 2 >"${ARTIFACT_ROOT}/logs/p3_r2_gpu2.log" 2>&1 & PID2=$!

python -m filament.train_refiner --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --run-name p3_r3_unetpp_b3_cldice --manifest "${MANIFEST}" --architecture unetplusplus --encoder efficientnet-b3 \
  --crop-size 512 --batch 2 --epochs 30 --skeleton-weight 0.2 --device 3 >"${ARTIFACT_ROOT}/logs/p3_r3_gpu3.log" 2>&1 & PID3=$!

echo "R0 PID=${PID0}; R1 PID=${PID1}; R2 PID=${PID2}; R3 PID=${PID3}"
wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"
echo "All P3 refiner ablations completed."
