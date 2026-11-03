# Solar Filament Instance Segmentation — P4 Fusion S2

Reproducible training and inference code for our best submitted system in the
[Solar Filament Segmentation 2026 Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026/overview).

The recorded submission was:

```text
submission_p4_fusion_s2.csv
Public PQ: 0.35
```

This repository contains source code and configuration only. Competition data,
generated datasets, model weights, prediction caches, logs, and submissions are
excluded from Git.

## Method

P4 Fusion S2 combines three independently trained branches:

1. **D3:** YOLOv8l-seg at 2048 px, trained on all annotation records with
   conservative augmentation.
2. **D4:** YOLOv8l-seg at 2048 px, trained using one medoid annotation record per
   physical image.
3. **S2:** U-Net++ with an EfficientNet-B3 encoder, radial-position input,
   centerline supervision, and Skeleton Recall Loss.

At inference time, D3 and D4 proposals are clustered and used as instance
seeds. S2 supplies dense foreground and centerline probabilities. Seeded
watershed separates instances, resolves overlaps, and removes small masks.

The submitted fixed post-processing configuration is:

```text
detector confidence       0.30
semantic threshold       0.55
minimum mask area         200 pixels
detector cluster IoU      0.45
ROI expansion             0.20
centerline boost          0.10
semantic tile size        768
semantic tile overlap     0.50
test detector size        2048
```

Historical grouped-validation result for this configuration:

```text
pooled/global PQ: 0.418111
TP: 1010, FP: 622, FN: 583
```

The Kaggle score is an observed result, not a guarantee. GPU kernels, package
versions, pretrained weights, and training nondeterminism may cause small
differences when retraining.

## Expected machine layout

The scripts use the current repository and an ignored local `artifacts/`
directory by default:

```text
${PROJECT_ROOT}                       cloned repository
${PROJECT_ROOT}/filament-segmentation-2026.zip
${ARTIFACT_ROOT}                      generated artifacts and checkpoints
```

The paths can be overridden when launching a script:

```bash
PROJECT_ROOT=/path/to/repo \
ARTIFACT_ROOT=/path/to/artifacts \
ZIP_PATH=/path/to/filament-segmentation-2026.zip \
bash scripts/prepare_reproduce_p4_s2.sh
```

Use a fresh artifact directory for a clean reproduction. The preparation code
does not overwrite an already materialized dataset.

## 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL> solar-filament26
cd solar-filament26
export PROJECT_ROOT="$(pwd)"
export ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts"
export ZIP_PATH="${PROJECT_ROOT}/filament-segmentation-2026.zip"
```

## 2. Download and place the competition data

Accept the competition rules on Kaggle, download
`filament-segmentation-2026.zip`, and place it at:

```text
${PROJECT_ROOT}/filament-segmentation-2026.zip
```

You may use the Kaggle CLI after configuring your Kaggle credentials:

```bash
python -m pip install kaggle
kaggle competitions download \
  -c filament-segmentation-2026 \
  -p ${PROJECT_ROOT}
```

Do not commit the archive or its extracted contents. The preparation command
extracts it under the external artifact root. The archive must contain:

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│   └── train_images/*.jpeg
└── test/
    └── test_images/*.jpeg
```

Validate that the archive is not truncated:

```bash
unzip -t filament-segmentation-2026.zip
```

## 3. Create the Python environment

Python 3.11 and an NVIDIA GPU are recommended. On the original server, the
working environment is named `finetune-sd`:

```bash
conda create -n finetune-sd python=3.11 -y
conda activate finetune-sd
```

Install a CUDA-enabled PyTorch build compatible with the installed NVIDIA
driver first. PyTorch is intentionally not pinned in `requirements.txt`, since
the correct wheel depends on the host driver. Confirm that this succeeds before
continuing:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
PY
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip wheel
python -m pip install --force-reinstall \
  numpy==1.26.4 opencv-python-headless==4.11.0.86
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
export PYTHONPATH=${PROJECT_ROOT}/src
```

Run the checks:

```bash
python -m compileall -q src
pytest -q
```

## 4. Reproduce the exact grouped split and datasets

The winning run used `GroupShuffleSplit(test_size=0.20, random_state=3020)`
over physical filenames. It is not equivalent to randomly splitting a list of
unique filenames. The dedicated preparation script reproduces the original
142-image/233-annotation-record validation set and then creates both D3 and D4
training datasets:

```bash
mkdir -p ${ARTIFACT_ROOT}/logs
bash scripts/prepare_reproduce_p4_s2.sh \
  2>&1 | tee ${ARTIFACT_ROOT}/logs/p4_s2_prepare.log
```

Expected datasets:

```text
${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020
${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020
```

The script stops if the validation split is not exactly 142 physical images
and 233 annotation records.

## 5. Train the three components

The detector chain must be trained in this order:

```text
generic yolov8l-seg.pt
        ↓
1536 px grouped baseline, 60 epochs
        ↓
D3 and D4 at 2048 px, 30 epochs
```

S2 is trained in parallel with D3/D4 after the baseline completes. The script
uses GPU 0 for the baseline, then GPU 0/1/2 for D3/D4/S2:

```bash
bash scripts/train_reproduce_p4_s2.sh
```

Monitor training:

```bash
watch -n 2 nvidia-smi
tail -f ${ARTIFACT_ROOT}/logs/p2_d3_yolov8l_2048_conservative.log
tail -f ${ARTIFACT_ROOT}/logs/p2_d4_yolov8l_2048_medoid.log
tail -f ${ARTIFACT_ROOT}/logs/p3_s2_semantic_skeleton_recall.log
```

Required checkpoints:

```text
${ARTIFACT_ROOT}/runs/yolov8l_seg_grouped_baseline/weights/best.pt
${ARTIFACT_ROOT}/runs/p2_d3_yolov8l_2048_conservative/weights/best.pt
${ARTIFACT_ROOT}/runs/p2_d4_yolov8l_2048_medoid/weights/best.pt
${ARTIFACT_ROOT}/runs/p3_s2_semantic_skeleton_recall/best.pt
```

Training intentionally refuses to overwrite an existing run directory. Use a
fresh artifact root or archive an old run before reproducing.

## 6. Evaluate the fixed P4-S2 configuration

This command first builds low-confidence D3/D4 validation caches and then
evaluates only the fixed submitted fusion parameters. It does not perform a new
hyperparameter search:

```bash
bash scripts/evaluate_reproduce_p4_s2.sh
```

Inspect the recorded configuration and score:

```bash
cat ${ARTIFACT_ROOT}/results/p4_fusion_s2/best_fusion.json
```

The official metric used for model selection is pooled/global Panoptic Quality
with strict matching at `IoU > 0.5`.

## 7. Generate the test submission

```bash
bash scripts/infer_best_fusion.sh p4_fusion_s2 0
```

Output:

```text
${ARTIFACT_ROOT}/results/p4_fusion_s2/submission_p4_fusion_s2.csv
```

Audit the file before uploading:

```bash
python - <<'PY'
from pathlib import Path
import os
import pandas as pd

path = Path(os.environ["ARTIFACT_ROOT"]) / "results/p4_fusion_s2/submission_p4_fusion_s2.csv"
frame = pd.read_csv(path)
assert list(frame.columns) == ["filament_id", "segmentation_rle"]
assert frame["filament_id"].is_unique
assert frame["segmentation_rle"].notna().all()
print(frame.shape)
print(frame.head())
PY
```

Upload with the Kaggle web UI or CLI:

```bash
kaggle competitions submit \
  -c filament-segmentation-2026 \
  -f ${ARTIFACT_ROOT}/results/p4_fusion_s2/submission_p4_fusion_s2.csv \
  -m "P4 Fusion S2: D3+D4 detector seeds plus Skeleton Recall semantic branch"
```

## Repository structure

```text
configs/                  experiment configurations
docs/                     research notes and extended runbooks
notebooks/                imported/reference notebooks
scripts/                  preparation, training, evaluation, and inference entry points
src/filament/             Python package
tests/                    metric, split, fusion, and data tests
requirements.txt          non-PyTorch Python dependencies
pyproject.toml            installable package metadata
```

## Important reproducibility notes

- Multiple annotation records may reference the same physical solar image.
  Splitting by COCO `image_id` leaks identical pixels across train and
  validation. All reproduction splits are grouped by `file_name`.
- The P4-S2 score of 0.35 is the highest public score observed for this project
  so far. Oracle results around 0.61 use validation ground truth for proposal
  selection and are **not** leaderboard results.
- The test set has no public ground truth. `evaluate` scores the held-out train
  split; `infer` creates a Kaggle submission from the competition test images.
- Keep all generated artifacts outside the Git repository. See `.gitignore`.

## Extended experiments

The five-fold OOF proposal bank, SIRA actual-seed refiner, neighbor-exclusion
ablation, and quality calibration are documented in
[`docs/P0_P1_M1_M4_RUNBOOK.md`](docs/P0_P1_M1_M4_RUNBOOK.md). They are research
experiments and are not the submitted P4-S2 pipeline described above.
