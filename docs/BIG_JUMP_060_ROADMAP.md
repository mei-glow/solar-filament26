# Roadmap toward a legitimate 0.60 public PQ

Date: 2026-10-29  
Current best public PQ: **0.35 (P4-S2)**  
Current best local pooled/global PQ: **0.42410 (P4-S1-global)**

## 1. Feasibility statement

The requested public target of 0.60 is a stretch target, not a promised result.
The competition organizers state that post-rescore PQ above 0.35 is already
valuable. Older scores around 0.55-0.69 may be from the previous metric or
non-reproducible pipelines and must not be used as a target without confirming
that they are legitimate post-August-2026 PQ results.

The first hard gate is an oracle study. If the current proposal bank cannot
reach at least 0.65 pooled PQ even when validation GT is used to select/repair
proposals, then 0.60 cannot be reached by a mask refiner alone. Detection and
proposal recovery must change first.

## 2. What the current numbers imply

For S1-global:

```text
TP = 1020, FP = 610, FN = 573
denominator = 1020 + 0.5*610 + 0.5*573 = 1611.5
global RQ = 1020 / 1611.5 = 0.633
global SQ = global PQ / global RQ = 0.4241 / 0.633 ~= 0.670
```

At the current SQ, reaching PQ 0.60 would require:

```text
required RQ = 0.60 / 0.670 ~= 0.896
```

Therefore the main bottleneck is not only boundary quality. The system must
convert many current FP/FN pairs into correctly matched instances while
preserving an IoU above the strict 0.5 threshold. A better semantic backbone
alone is unlikely to do this. S3, S4 and S5 already provide negative evidence:
topology loss, Frangi input and iterative residual segmentation all reduced PQ.

## 3. Priority ladder

| Priority | Experiment | Main question | Train cost | Promotion gate |
|---:|---|---|---:|---|
| P0 | Exact official evaluator + global rerank | Are model-selection decisions metric-aligned? | none | exact agreement with official notebook |
| P1 | Proposal-oracle audit | Are GT instances recoverable from current proposals? | none | report mask/box recall and oracle PQ |
| P2 | Conservative global-PQ sweep | Can cached predictions reduce FP cheaply? | none | +0.005 pooled PQ |
| M1 | Five-fold OOF proposal bank | Can a refiner/quality model be trained without leakage? | high | complete OOF candidate coverage |
| M2 | SIRA-R0 actual-seed refiner | Can poor-IoU candidates cross IoU 0.5? | medium | +0.015 pooled OOF PQ |
| M3 | SIRA-R1 neighbor exclusion | Can merge/FP be reduced? | medium | +0.005 PQ and >=10% fewer merge/FP errors |
| M4 | OOF quality calibration | Can unreliable refined masks be removed? | low | +0.010 PQ or material FP reduction without recall collapse |
| M5 | Global-local tiled detector | Are small/missed filaments recoverable? | high, conditional | +0.015 PQ or >=10% FN reduction |
| M6 | Mask DINO/Mask2Former smoke run | Does set prediction solve proposal/merge failures better? | high, conditional | +0.010 after bounded smoke run |
| M7 | Annotator-aware supervision | Does label sampling improve robustness? | medium | consistent gain on >=3 folds |
| M8 | Fold ensemble + full-data retrain | Convert selected method into final test model | high | OOF gain stable; submission audit passes |

Gains are gates, not additive promises.

## 4. P0: lock the exact official metric

The official self-evaluation notebook pools matched IoU, TP, FP and FN across
all annotator-image records, then calculates one PQ. The default selection
metric must therefore be `global_pq`, not the macro average `mean_pq`.

Required tests:

1. Empty prediction, empty GT and normal cases.
2. Strict `IoU > 0.5`, not `>= 0.5`.
3. Multiple annotators for one physical image.
4. Split, merge and overlapping-mask edge cases.
5. Byte-for-byte comparison of statistics with the official notebook on the
   same validation predictions.

No new training result is trusted until this passes.

## 5. P1: proposal-oracle audit

Use cached D3 and D4 predictions at base confidence 0.05. This is validation
diagnosis only; GT must never affect test predictions.

### Outputs per GT instance

- best detector box IoU;
- best detector mask IoU;
- best confidence and source model;
- area, bounding-box dimensions, elongation and disk radius;
- whether multiple proposals cover it;
- whether it participates in a split or merge.

### Required aggregate diagnostics

```text
proposal recall at box IoU > 0.1, 0.3, 0.5
proposal recall at mask IoU > 0.1, 0.3, 0.5
best-mask-IoU histogram
oracle selection PQ using existing masks
perfect-mask oracle PQ for proposal-covered GT
breakdown by area, thickness proxy, radial position and annotator agreement
```

### Decision gate

- Perfect-mask proposal oracle >= 0.65 and box recall >= 90%: prioritize SIRA.
- Oracle 0.50-0.65 or recall 75-90%: run SIRA and tiled proposal recovery in
  parallel.
- Oracle < 0.50 or recall < 75%: detector/proposal recovery is the first
  bottleneck; do not spend all GPUs on refinement.

## 6. P2: cheap pooled-PQ post-processing sweep

Use S1 and S2 cached probabilities. Optimize pooled/global PQ only.

First sweep:

```text
detector confidence: 0.275, 0.30, 0.325, 0.35, 0.40
semantic threshold: 0.50, 0.55, 0.60, 0.65, 0.70
minimum area: 150, 200, 250, 300, 400, 600
centerline boost: 0.00, 0.05, 0.10
rescue: disabled
```

Then tune geometry only around the winner:

```text
cluster IoU: 0.35, 0.45, 0.55
ROI expansion: 0.10, 0.20, 0.30
```

Do not jointly search all dimensions. Promote only a setting improving global
PQ by at least 0.005 with a plausible TP/FP/FN trade-off.

## 7. M1: build a leakage-safe OOF proposal bank

Actual detector seeds are required to train an instance-conditioned refiner.
Seeds synthesized from GT masks cause train-inference mismatch.

Train grouped detectors for five folds, always starting from generic pretrained
weights rather than a checkpoint that has seen the held-out fold. Four folds can
run concurrently; run the fifth when one GPU is free. Store for every OOF image:

```text
candidate mask and box
detector confidence
source model/scale/tile
best matched GT instance and IoU (training metadata only)
neighbor candidate masks
semantic S2 probability crop
```

The bank must include low-confidence candidates and hard false positives, not
only exported predictions.

## 8. M2: SIRA-R0 actual-seed instance refiner

### Training unit

One OOF detector proposal, not one GT crop.

### Native-resolution input channels

```text
1. normalized H-alpha intensity
2. CLAHE/local contrast
3. actual detector seed mask
4. candidate-distance/seed-affinity map
5. S2 semantic region probability
```

Use 512 or 768 native-resolution windows. Long instances are covered by
overlapping windows with the same instance identity; do not resize an entire
long bounding box to 256 pixels.

### Targets and sampling

- Positive: proposal matched to exactly one GT instance.
- Hard positive: box overlap exists but mask IoU is 0.1-0.5.
- Hard negative: high-confidence dark proposal with no GT match.
- Ambiguous examples are retained with lower supervision weight.

### Initial loss

```text
BCE + Dice
```

Do not add clDice initially. S3 shows that connectivity gains can create merges
and FP. Select checkpoints by end-to-end pooled PQ after reconstruction, not ROI
Dice.

## 9. M3: SIRA-R1 explicit neighbor exclusion

Add a sixth input channel containing competing detector seeds. For a target
instance, pixels belonging to other matched GT instances receive an exclusion
penalty:

```text
loss = BCE + Dice + lambda_exclusion * predicted_probability_on_neighbor_GT
```

This is the central research ablation. It directly targets the failure mode in
which an image-only crop refiner absorbs two or more nearby dark filaments.

Stop if merge rate or pooled PQ does not improve over SIRA-R0.

## 10. M4: OOF mask-quality calibration

Train a small calibrated classifier/regressor on OOF refined candidates. The
target is either max GT IoU or `IoU > 0.5`. Candidate features include:

```text
detector confidence
predicted/refined mask agreement
D3/D4 support count
mean and quantiles of refiner probability
semantic support
area, elongation and component count
boundary uncertainty
neighbor overlap/competition
disk radius
```

Start with logistic regression or gradient boosting before a neural head. Tune
the export threshold directly on pooled OOF PQ. This stage is expected to be
more useful than another semantic backbone because current public results favor
the lower-FP S2 configuration.

## 11. M5: conditional global-local proposal recovery

Run only if P1 shows insufficient proposal coverage, especially for small GT.

### Detector views

- full disk at 2048;
- overlapping 1024 native-resolution tiles with context;
- optional 1536 full-disk diversity view.

Train tiles with aggressive hard-negative sampling from sunspots, limb artifacts
and unlabeled dark regions. Merge candidates across views before refinement;
never export tile detections directly without global de-duplication.

Promotion requires a real FN reduction after quality calibration, not just more
raw proposals.

## 12. M6: conditional set-prediction model

Mask DINO and Mask2Former are relevant because they predict a set of instance
masks and can model global competition between instances. Mask DINO combines a
detection framework with a mask branch; Mask2Former uses masked attention for
universal segmentation. However, both are higher-risk than SIRA on only 707
physical images and require careful high-resolution tiling/pretraining.

Run one bounded smoke experiment only if:

- proposal oracle says current YOLO coverage/merge behavior is the bottleneck;
- SIRA cannot compensate;
- a pretrained R50 checkpoint and stable implementation are available.

Use 1024 crops/tiles, one class, grouped fold 0, and a 10-15 epoch gate. Stop if
proposal recall or pooled PQ does not exceed D3/S2 by at least 0.01.

References:

- Mask DINO: https://arxiv.org/abs/2206.02777
- Mask2Former: https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html
- Skeleton Recall Loss: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09904.pdf

## 13. M7: annotator-aware supervision

Compare only after SIRA-R0 works:

1. medoid annotation set;
2. random annotation set per physical image per epoch;
3. uncertainty weighting from annotator agreement.

Do not union independent instance sets. For the leaderboard, select by grouped
OOF pooled PQ. For the paper, report results per annotator and by agreement.

## 14. M8: fold ensemble and full-data training

After all architecture and thresholds are frozen:

1. Keep the five grouped fold models and ensemble their test candidates/masks.
2. Train one additional detector/refiner on all 707 physical images.
3. Use OOF-derived calibration and thresholds unchanged.
4. Cluster candidates across folds/views, require support for weak singletons,
   average mask probability within clusters and resolve overlap by calibrated
   instance score.
5. Compare fold ensemble, full-data model and their combination locally using
   OOF simulations before using Kaggle submissions.

## 15. Four-GPU execution schedule

### Days 1-2: measurement and cheap wins

- CPU/GPU 0: exact evaluator parity and global rerank.
- CPU: proposal oracle and error taxonomy.
- GPU 1-3: finish only already-running inference; no new semantic architectures.
- Submit S1-global only if its CSV/config audit passes.

### Days 2-5: OOF proposal bank

- GPU 0-3: detector folds 0-3.
- First free GPU: fold 4.
- Generate OOF predictions at low confidence and build candidate bank.

### Days 5-8: core new method

- GPU 0: SIRA-R0, ResNet34 U-Net.
- GPU 1: SIRA-R0, U-Net++ B3 capacity ablation.
- GPU 2: SIRA-R1 neighbor exclusion.
- GPU 3: tiled detector only if oracle requires it; otherwise annotator-random
  SIRA ablation.

### Days 8-10: calibration and recovery

- CPU: OOF quality model and threshold calibration.
- GPU 0-1: reconstruct/evaluate best SIRA variants.
- GPU 2: tiled detector continuation if promoted.
- GPU 3: bounded Mask DINO/Mask2Former smoke run only if gated in.

### Days 10-13: confirmation

- Run the best configuration on remaining grouped folds.
- Freeze every hyperparameter.
- Reject gains that exist on only one fold or one annotator.

### Days 13-16: final models

- GPU 0-1: full-data detector(s).
- GPU 2-3: full-data semantic/refiner(s).
- Generate fold ensemble and full-data submissions.

### Final day

- Submission schema/RLE/non-overlap audit.
- Select at most two scientifically supported final submissions.
- Freeze logs, configs, checkpoints and paper tables.

## 16. Submission discipline and stop rules

- No Kaggle submission without >=0.005 pooled OOF improvement or a deliberate
  precision/recall probe.
- Stop semantic-only architecture exploration; S3-S5 are sufficient negative
  ablations.
- Stop SIRA if proposal-conditioned refinement does not add 0.015 pooled PQ.
- Stop topology loss if merge/FP increases.
- Stop tiled detection if extra proposals do not become matched TP after quality
  calibration.
- Stop transformer experiments after the bounded smoke gate.
- Do not claim a path to 0.60 unless the oracle ceiling is >=0.65 and gains hold
  on multiple grouped folds.

## 17. Expected outcomes

Reasonable, evidence-based tiers are:

```text
near-term cached tuning / S1-global: 0.35-0.37 public
strong SIRA + calibration + full data: roughly 0.40-0.50 is a defensible stretch
0.60: possible only if oracle coverage is very high and current FP/FN are mostly
      recoverable matching failures rather than true detector misses/domain shift
0.70-0.80: no current post-rescore evidence supports this target
```

These are planning ranges, not guaranteed scores. The oracle audit decides
whether the 0.60 target remains technically credible.
