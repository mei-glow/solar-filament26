# Server runbook — P0 to P3 on four GPUs

All commands below run on the Linux server in the existing `finetune-sd` Conda environment.

## 0. Install/update the code

```bash
cd ${PROJECT_ROOT}
conda activate finetune-sd
bash setup_server.sh
export PYTHONPATH=${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
pytest -q
```

Verify four GPUs:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

## 1. P0 — Build safe manifests/datasets

The existing baseline checkpoint must keep its original split. The script therefore requires and preserves:

```text
${ARTIFACT_ROOT}/results/split_manifest.csv
```

Build legacy-safe P2 datasets plus clean five-fold paper datasets:

```bash
bash scripts/prepare_p0.sh 2>&1 | tee ${ARTIFACT_ROOT}/logs/p0_prepare.log
```

Expected outputs:

```text
${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/
${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/
${ARTIFACT_ROOT}/datasets/fold0_medoid_s3020/
${ARTIFACT_ROOT}/datasets/fold0_maxcomplete_4class_s42/
${ARTIFACT_ROOT}/datasets/fold0_randomsingle_s3020/
${ARTIFACT_ROOT}/results/fold_manifest_s3020.csv
${ARTIFACT_ROOT}/results/fold_manifest_s42.csv
```

Each physical filename must occur in exactly one split/fold. Validation retains all annotators except the explicit public-notebook reproduction dataset.

## 2. P1 — Audit the existing baseline

Run cached official-PQ evaluation, error taxonomy and visual gallery:

```bash
BASE=${ARTIFACT_ROOT}/runs/yolov8l_seg_grouped_baseline/weights/best.pt
MANIFEST=${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/manifest.csv

python -m filament.evaluate \
  --run-name p1_baseline_audit \
  --checkpoint "$BASE" \
  --manifest "$MANIFEST" \
  --imgsz 1536 \
  --device 0

python -m filament.analysis.error_taxonomy \
  --run-name p1_baseline_audit \
  --checkpoint "$BASE" \
  --manifest "$MANIFEST" \
  --imgsz 1536

python -m filament.analysis.visualize_errors \
  --run-name p1_baseline_audit \
  --checkpoint "$BASE" \
  --manifest "$MANIFEST" \
  --imgsz 1536

python -m filament.analysis.stratified_metrics \
  --run-name p1_baseline_audit \
  --checkpoint "$BASE" \
  --manifest "$MANIFEST" \
  --imgsz 1536
```

Outputs:

```text
${ARTIFACT_ROOT}/results/p1_baseline_audit/postprocess_tuning.csv
${ARTIFACT_ROOT}/results/p1_baseline_audit/metrics_by_annotation.csv
${ARTIFACT_ROOT}/results/p1_baseline_audit/error_taxonomy_summary.json
${ARTIFACT_ROOT}/results/p1_baseline_audit/worst_predictions_gallery.jpg
${ARTIFACT_ROOT}/results/p1_baseline_audit/metrics_by_size.csv
```

### PQ sweep all saved baseline epochs on four GPUs

```bash
python -m filament.sweep_checkpoints \
  --checkpoint-dir ${ARTIFACT_ROOT}/runs/yolov8l_seg_grouped_baseline/weights \
  --manifest ${ARTIFACT_ROOT}/datasets/legacy_duplicate_s3020/manifest.csv \
  --prefix p1_d1_baseline \
  --devices 0,1,2,3 \
  --workers 4 \
  --imgsz 1536 \
  --top-k 3
```

The sweep uses a coarse grid. Rerun the top three with `filament.evaluate` without `--quick`; their cached inference is reused.

The public YOLO+U-Net notebook source and audit are stored at:

```text
notebooks/reference/yolo-u-net-solar-filament-segmentation.ipynb
docs/REFERENCE_NOTEBOOK_AUDIT.md
```

Its closest detector reproduction is GPU 3 in the P2 launcher.

## 3. P2 — Four detector experiments in parallel

Start all four runs:

```bash
bash scripts/run_p2_4gpu.sh
```

To use a different baseline checkpoint:

```bash
bash scripts/run_p2_4gpu.sh /absolute/path/to/best.pt
```

GPU assignment:

| GPU | Run | Purpose |
|---:|---|---|
| 0 | `p2_d2_yolov8l_2048_standard` | 1536→2048 fine-tune, standard augmentation |
| 1 | `p2_d3_yolov8l_2048_conservative` | 2048, no mosaic/translate/mixup |
| 2 | `p2_d4_yolov8l_2048_medoid` | 2048, medoid training labels |
| 3 | `p2_ref_yolo11s_1024_4class` | reproduce public 0.69 detector recipe |

Monitor:

```bash
watch -n 2 nvidia-smi
tail -f ${ARTIFACT_ROOT}/logs/p2_d2_gpu0.log
tail -f ${ARTIFACT_ROOT}/logs/p2_d3_gpu1.log
tail -f ${ARTIFACT_ROOT}/logs/p2_d4_gpu2.log
tail -f ${ARTIFACT_ROOT}/logs/p2_ref_gpu3.log
```

The trainer refuses to overwrite an existing run. Resume an interrupted run explicitly:

```bash
python -m filament.train --run-name p2_d2_yolov8l_2048_standard --resume
```

After all four finish, evaluate, classify errors and render galleries in parallel:

```bash
bash scripts/evaluate_p2_4gpu.sh
```

Generate four candidate submissions only after evaluation has selected each
run's post-processing parameters:

```bash
bash scripts/infer_p2_4gpu.sh
```

Each CSV is written under
`${ARTIFACT_ROOT}/results/<run-name>/submission_<run-name>.csv` with a
sidecar metadata file containing checkpoint and submission SHA-256 hashes.

Compare:

```bash
for run in \
  p2_d2_yolov8l_2048_standard \
  p2_d3_yolov8l_2048_conservative \
  p2_d4_yolov8l_2048_medoid \
  p2_ref_yolo11s_1024_4class
do
  echo "===== $run ====="
  cat "${ARTIFACT_ROOT}/results/$run/best_postprocess.json"
  cat "${ARTIFACT_ROOT}/results/$run/error_taxonomy_summary.json"
done
```

Do not compare the 4-class reproduction's local PQ directly to the other three: it intentionally uses the public notebook's one-annotator validation protocol. It tests the reported leaderboard recipe, not paper evidence.

## 4. P3 — Four refiner ablations in parallel

```bash
bash scripts/run_p3_4gpu.sh
```

GPU assignment:

| GPU | Run | Architecture/loss |
|---:|---|---|
| 0 | `p3_r0_unet_r34_bce_dice` | U-Net ResNet34, BCE + Dice |
| 1 | `p3_r1_unetpp_b3_bce_dice` | U-Net++ EfficientNet-B3, BCE + Dice |
| 2 | `p3_r2_unetpp_b3_boundary` | R1 + boundary-weighted BCE |
| 3 | `p3_r3_unetpp_b3_cldice` | R1 + differentiable clDice |

Monitor:

```bash
watch -n 2 nvidia-smi
tail -f ${ARTIFACT_ROOT}/logs/p3_r0_gpu0.log
tail -f ${ARTIFACT_ROOT}/logs/p3_r1_gpu1.log
tail -f ${ARTIFACT_ROOT}/logs/p3_r2_gpu2.log
tail -f ${ARTIFACT_ROOT}/logs/p3_r3_gpu3.log
```

Check results:

```bash
for run in \
  p3_r0_unet_r34_bce_dice \
  p3_r1_unetpp_b3_bce_dice \
  p3_r2_unetpp_b3_boundary \
  p3_r3_unetpp_b3_cldice
do
  echo "===== $run ====="
  cat "${ARTIFACT_ROOT}/runs/$run/history.json"
done
```

ROI Dice/IoU only validates that the structure branch learned its crop task. A refiner is promoted only after P4 fusion increases official end-to-end PQ and does not increase merge errors.

## 5. Individual config-based commands

Every run also has a YAML config. Explicit CLI values override YAML values.

```bash
python -m filament.train \
  --config configs/p2_d3_yolov8l_2048_conservative.yaml \
  --model ${ARTIFACT_ROOT}/runs/yolov8l_seg_grouped_baseline/weights/best.pt

python -m filament.train_refiner \
  --config configs/p3_r1_unetpp_b3.yaml
```

## 6. If a run runs out of memory

- YOLO 2048 already uses batch 1. If it still fails, use `--imgsz 1792` for a separately named run; do not silently change an existing experiment.
- Refiner: lower `--batch`; keep crop size fixed so the experiment remains interpretable.
- clDice is the heaviest ablation and starts at batch 2.
- Never run two processes on the same GPU unless deliberately profiling memory headroom.
