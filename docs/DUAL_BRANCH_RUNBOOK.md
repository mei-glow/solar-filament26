# Dual-branch semantic/topology + instance detector runbook

This pipeline keeps the P2 D3/D4 YOLO instance detectors and adds a full-image
semantic/topology branch. Detector masks provide instance seeds. A semantic
probability field expands and repairs them with seeded watershed, clips each
instance to its detector neighborhood, removes overlap, and can optionally
rescue elongated semantic components that have no detector seed.

## Implemented ablations

| ID | Semantic branch | Structural idea | GPU in launch script |
|---|---|---|---:|
| S0 | U-Net/ResNet34 | region BCE + Dice control | 0 |
| S1 | U-Net++/EfficientNet-B3 | stronger semantic control | 1 |
| S2 | U-Net++/EfficientNet-B3 | centerline head + skeleton recall loss | 2 |
| S3 | U-Net++/EfficientNet-B3 | centerline + boundary heads + clDice | 3 |
| S4 | U-Net++/EfficientNet-B3 | Frangi vesselness + auxiliary structural losses | 0, wave 2 |
| S5 | iterative U-Net++ | residual second-stage expert + structural losses | 1, wave 2 |

All runs use the same legacy grouped validation manifest as D3/D4. This is
required for a valid detector/fusion comparison and prevents physical-image
leakage across train and validation.

## 0. One-time server setup

```bash
cd ${PROJECT_ROOT}
conda activate finetune-sd
bash setup_server.sh
export PYTHONPATH=${PROJECT_ROOT}/src
pytest -q
```

`setup_server.sh` installs segmentation-models-pytorch and scikit-image in the
existing `finetune-sd` environment. It does not replace PyTorch.

## 1. Train the first four models in four tmux sessions

```bash
cd ${PROJECT_ROOT}
bash scripts/launch_p3_semantic_tmux.sh
```

Monitor them with:

```bash
tmux ls
tmux attach -t p3_s0
# Detach without stopping: Ctrl-b, then d
watch -n 2 nvidia-smi
tail -f ${ARTIFACT_ROOT}/logs/p3_s3_gpu3.log
```

Checkpoints are saved as:

```text
${ARTIFACT_ROOT}/runs/<semantic-run-name>/best.pt
${ARTIFACT_ROOT}/runs/<semantic-run-name>/last.pt
```

If a 768-pixel U-Net++ run is out of memory, change only `batch: 2` to
`batch: 1` in that model's YAML and launch it again after moving the incomplete
run directory to a backup name.

## 2. Optional semantic-only measurements

This is for the paper (Dice, IoU, clDice, skeleton recall). It is not the
competition metric.

```bash
conda activate finetune-sd
cd ${PROJECT_ROOT}
bash scripts/evaluate_p3_semantic_4gpu.sh
```

Outputs:

```text
${ARTIFACT_ROOT}/results/<semantic-run-name>/semantic_tuning.csv
${ARTIFACT_ROOT}/results/<semantic-run-name>/best_semantic.json
```

## 3. Tune dual-branch fusion with official PQ

Run only after S0-S3 training has finished and the D3/D4 raw validation caches
still exist under `${ARTIFACT_ROOT}/predictions/raw/`.

```bash
cd ${PROJECT_ROOT}
bash scripts/launch_p4_fusion_tmux.sh
```

Results:

```text
${ARTIFACT_ROOT}/results/p4_fusion_s0/fusion_tuning.csv
${ARTIFACT_ROOT}/results/p4_fusion_s0/best_fusion.json
# likewise for s1, s2, s3
```

Quick comparison:

```bash
for run in p4_fusion_s0 p4_fusion_s1 p4_fusion_s2 p4_fusion_s3; do
  echo "========== ${run} =========="
  cat "${ARTIFACT_ROOT}/results/${run}/best_fusion.json"
done
```

Select by held-out `mean_pq`, not semantic Dice. The P2 D3 reference is
`mean_pq=0.38815`; a fusion run must beat it before spending a submission.

## 4. Create a submission from the best fusion run

Replace `p4_fusion_s2` with the selected run:

```bash
cd ${PROJECT_ROOT}
conda activate finetune-sd
bash scripts/infer_best_fusion.sh p4_fusion_s2 0
```

Submission location:

```text
${ARTIFACT_ROOT}/results/p4_fusion_s2/submission_p4_fusion_s2.csv
```

## 5. Second wave: vessel-specific ablations

Run S4/S5 only after the first wave so they do not compete for GPUs:

```bash
cd ${PROJECT_ROOT}
bash scripts/launch_p3_vessel_tmux.sh
```

After both finish, evaluate their official fusion PQ:

```bash
bash scripts/launch_p3_vessel_semantic_eval_tmux.sh
bash scripts/launch_p4_vessel_fusion_tmux.sh
```

The result names are `p4_fusion_s4` and `p4_fusion_s5`; inference uses the
same `infer_best_fusion.sh` command. To create both submissions concurrently:

```bash
bash scripts/launch_p4_vessel_infer_tmux.sh
```

## 6. Focused rescue and TTA ablations

Do these only for the best branch. Existing predictions are cached, so rescue
threshold tuning does not repeat semantic inference:

```bash
python -m filament.evaluate_fusion \
  --run-name p4_fusion_s2_rescue \
  --manifest ${ARTIFACT_ROOT}/datasets/legacy_medoid_s3020/manifest.csv \
  --semantic-run-name p3_s2_semantic_skeleton_recall \
  --detector-run-names p2_d3_yolov8l_2048_conservative,p2_d4_yolov8l_2048_medoid \
  --detector-checkpoints ${ARTIFACT_ROOT}/runs/p2_d3_yolov8l_2048_conservative/weights/best.pt,${ARTIFACT_ROOT}/runs/p2_d4_yolov8l_2048_medoid/weights/best.pt \
  --device 0 --semantic-batch 2 \
  --detector-conf-values 0.20,0.25,0.30 \
  --semantic-threshold-values 0.35,0.45,0.55 \
  --min-area-values 100,200 \
  --rescue-threshold-values none,0.75,0.85
```

For TTA, add `--tta` and use a new run name. TTA has a separate cache and
therefore repeats semantic inference once.

## Interpretation

- High FP: raise detector confidence/minimum area; keep rescue disabled.
- High FN: lower detector confidence or test conservative rescue.
- Good RQ but weak SQ: semantic boundary/threshold is the likely bottleneck.
- Good semantic Dice but flat PQ: instance seeding/separation, not region
  segmentation, is the bottleneck.
- Always record both local PQ and Kaggle score. A leaderboard score alone is
  not evidence for the paper because the private test labels are unavailable.
