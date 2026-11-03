# P0-P1-M1-M4 execution runbook

All commands use the server environment and write durable artifacts to
`${ARTIFACT_ROOT}`. Do not skip a gate simply to start the next stage.

## One-time setup after syncing the repository

```bash
cd ${PROJECT_ROOT}
conda activate finetune-sd
export PYTHONPATH=${PROJECT_ROOT}/src
python -m compileall -q src
pytest -q
```

Expected: all tests pass, including strict `IoU > 0.5` metric tests.

## P0: metric lock

New evaluator runs select `global_pq`, not `mean_pq`. P0 does not train a model.

## P1: D3+D4 proposal oracle

The D3/D4 caches must have `base_conf=0.05`, `imgsz=2048`, `nms_iou=0.70`,
and `max_det=100`.

```bash
bash scripts/run_p1_oracle.sh 2>&1 | tee ${ARTIFACT_ROOT}/logs/p1_oracle.log
cat ${ARTIFACT_ROOT}/results/p1_proposal_oracle_d3_d4/proposal_oracle.json
```

Prioritize refinement only if `perfect_oracle_global_pq >= 0.65` and box recall
is high. P1 is validation diagnosis and never changes test predictions.

## M1: five-fold OOF detector and semantic bank

```bash
bash scripts/prepare_m1_folds.sh

bash scripts/launch_m1_detectors_tmux.sh 0 1 2 3
# after folds 0-3 finish
bash scripts/launch_m1_detectors_tmux.sh 4

bash scripts/launch_m1_semantic_tmux.sh 0 1 2 3
# after folds 0-3 finish
bash scripts/launch_m1_semantic_tmux.sh 4

bash scripts/launch_m1_oof_cache_tmux.sh 0 1 2 3
# after folds 0-3 finish
bash scripts/launch_m1_oof_cache_tmux.sh 4
```

Verify exactly 707 physical images before M2:

```bash
find ${ARTIFACT_ROOT}/predictions/oof/m1_oof_yolov8l -name '*.json.gz' | wc -l
find ${ARTIFACT_ROOT}/predictions/semantic_oof/m1_oof_s2 -name '*.npz' | wc -l
```

Both counts must be 707. Each image is predicted only by models that excluded
that physical image from training.

## M2: SIRA-R0 actual-seed refiner

```bash
bash scripts/launch_sira_tmux.sh m2 0 1 2 3
# after folds 0-3 finish
bash scripts/launch_sira_tmux.sh m2 4

bash scripts/launch_sira_eval_tmux.sh m2 0 1 2 3
# after folds 0-3 finish
bash scripts/launch_sira_eval_tmux.sh m2 4

for fold in 0 1 2 3 4; do
  echo "===== M2 fold ${fold} ====="
  cat "${ARTIFACT_ROOT}/results/m2_sira_oof_fold${fold}/best_sira.json"
done
```

M2 uses native 512 windows and overlapping windows for long proposals. No seed
is synthesized from GT.

## M3: neighbor-exclusion SIRA

M3 is a separate five-fold experiment, not a continuation checkpoint from M2.

```bash
bash scripts/launch_sira_tmux.sh m3 0 1 2 3
# after folds 0-3 finish
bash scripts/launch_sira_tmux.sh m3 4

bash scripts/launch_sira_eval_tmux.sh m3 0 1 2 3
# after folds 0-3 finish
bash scripts/launch_sira_eval_tmux.sh m3 4
```

M3 adds competing proposal seeds as channel 6 and uses
`lambda_exclusion=0.25`. Reject it if pooled PQ/FP/merge errors do not improve.

## M4: leakage-safe quality calibration

Calibration uses group cross-validation by physical filename. Reported scores
come from cross-fitted probabilities, never in-sample probabilities.

```bash
bash scripts/run_m4_calibration.sh m2 | tee ${ARTIFACT_ROOT}/logs/m4_quality_m2.log
bash scripts/run_m4_calibration.sh m3 | tee ${ARTIFACT_ROOT}/logs/m4_quality_m3.log

cat ${ARTIFACT_ROOT}/results/m4_quality_m2/best_quality.json
cat ${ARTIFACT_ROOT}/results/m4_quality_m3/best_quality.json
```

Promote only a branch whose pooled OOF gain holds across multiple folds. These
are research/selection measurements, not submission CSVs. Test inference and a
submission are built only after freezing the winning M2/M3/M4 configuration.

## Monitoring

```bash
tmux ls
tail -f ${ARTIFACT_ROOT}/logs/<run>.log
nvidia-smi
```
