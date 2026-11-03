#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${ZIP_PATH:-${PROJECT_ROOT}/filament-segmentation-2026.zip}"
PYTHON="${PYTHON:-$(command -v python)}"
MASTER_MANIFEST="${ARTIFACT_ROOT}/results/p4_s2_split_manifest.csv"
export ARTIFACT_ROOT

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
[[ -f "${ZIP_PATH}" ]] || { echo "Missing competition archive: ${ZIP_PATH}" >&2; exit 1; }

"${PYTHON}" -m filament.prepare \
  --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-name legacy_duplicate_s3020 --annotation-strategy duplicate \
  --legacy-group-shuffle-split --seed 3020 --export-manifest "${MASTER_MANIFEST}"

"${PYTHON}" -m filament.prepare \
  --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" \
  --dataset-name legacy_medoid_s3020 --annotation-strategy medoid \
  --input-manifest "${MASTER_MANIFEST}"

"${PYTHON}" - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["ARTIFACT_ROOT"])
manifest = pd.read_csv(root / "results/p4_s2_split_manifest.csv")
assert manifest.groupby("file_name")["split"].nunique().max() == 1
summary = manifest.groupby("split").agg(records=("image_id", "size"), physical_images=("file_name", "nunique"))
print(summary)
assert int(summary.loc["val", "physical_images"]) == 142
assert int(summary.loc["val", "records"]) == 233
print("Exact P4-S2 legacy split verified.")
PY
