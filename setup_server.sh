#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"

cd "${PROJECT_ROOT}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "finetune-sd" ]]; then
  echo "ERROR: activate the existing Conda environment first:" >&2
  echo "  conda activate finetune-sd" >&2
  exit 1
fi

echo "Using Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(command -v python)"
python -m pip install --upgrade pip wheel
python -m pip install --force-reinstall "numpy==1.26.4" "opencv-python-headless==4.11.0.86"
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

mkdir -p "${ARTIFACT_ROOT}/logs"
test -w "${ARTIFACT_ROOT}" || {
  echo "ERROR: ${ARTIFACT_ROOT} is not writable by $(whoami)" >&2
  exit 1
}

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))
else:
    raise SystemExit("ERROR: PyTorch cannot initialize CUDA in finetune-sd")
PY

echo "Server environment is ready."
