# Expert strategy request: Solar Filament Segmentation Challenge 2026

## How I want you to help

Act as a senior computer-vision researcher and Kaggle competition strategist.
Read this brief critically, verify current competition details from official
sources, and propose a technically valid, reproducible plan to maximize the
current **Panoptic Quality (PQ)** score while also producing a strong machine
learning course paper.

My current public score is **0.35**. My stretch target is **0.70-0.80**, but do
not assume that target is feasible under the current re-scored PQ metric.
Explicitly assess whether it is realistic. The organizers have stated that a
PQ above 0.35 is already valuable, and older notebook titles/scores may predate
the August 2026 metric change. Do not recommend leaderboard exploits, test-label
leakage, manual test annotation, or metric hacking.

I need a concrete experimental roadmap, not a generic list of models.

## Official competition

- Competition: Solar Filament Segmentation Challenge 2026
- URL: https://www.kaggle.com/competitions/filament-segmentation-2026/overview
- Domain: individual solar-filament instance segmentation from GONG H-alpha
  full-disk observations.
- Deadline: 2026-11-15.
- Input at inference: only competition test H-alpha images.
- Output: one CSV row per predicted filament with columns `filament_id` and
  COCO compressed `segmentation_rle`; all images are 2048 x 2048.
- The metric was changed/re-scored in August 2026. Current primary metric is
  Panoptic Quality:

  `PQ = sum IoU(matched GT, prediction) / (TP + 0.5 FP + 0.5 FN)`

- Matching requires sufficient instance overlap (the official self-evaluation
  notebook is treated as the executable specification).
- The metric penalizes missed filaments, false filaments, fragmentation and
  over-merging. Region Dice alone is not the target.
- Public leaderboard uses only part of the hidden test set and displays two
  decimal places, so `0.34` and `0.35` are coarsely rounded signals.
- Organizers also assess qualitative morphology, pipeline description and code
  quality; source code must eventually be public.
- Official note: the organizers explicitly wrote that PQ > 0.35 is valuable
  and warned against hacking the evaluation. Treat old claims of 0.55-0.69
  cautiously unless confirmed after the PQ re-score.

## Dataset statistics

The complete official training annotation JSON was parsed locally.

| Statistic | Value |
|---|---:|
| Unique physical training images | 707 |
| COCO image/annotation records | 1,154 |
| Instance annotations | 8,199 |
| Test images | 180 |
| Image size | 2048 x 2048 for every record |
| Physical images with one annotation set | 411 |
| Physical images with two annotation sets | 145 |
| Physical images with three annotation sets | 151 |
| Physical images with multiple annotators | 296 |
| Instances per annotation record | min 1, median 7, mean 7.10, max 26 |

There are four chirality categories: Left, Right, Unidentifiable and Ambiguous.
The competition submission is class-agnostic: every filament is an instance.

Multiple COCO `image_id` records can reference the same physical `file_name`.
Therefore, splitting by COCO image ID leaks identical pixels between train and
validation. All valid local splits group by physical `file_name`.

The current development split contains approximately 565 physical training
images and 142 physical validation images. The 142 validation images correspond
to 233 annotation records and 1,593 GT instances because independent annotators
are evaluated separately.

## Compute and implementation constraints

- Linux server with four CUDA GPUs.
- Python/Conda environment: `finetune-sd`.
- PyTorch, Ultralytics, segmentation-models-pytorch, OpenCV, pycocotools and
  scikit-image.
- Detector images are 2048 x 2048; semantic models use overlapping 512/768
  tiles and reconstruct a full-resolution probability field.
- Raw detector and semantic validation predictions are cached, allowing many
  post-processing sweeps without repeating network inference.
- All checkpoints/results are stored outside the source tree with immutable run
  names and metadata.
- Local evaluator reproduces the official PQ matching logic and reports mean
  per-annotation PQ, global PQ, SQ, RQ, TP, FP and FN.

## Data and evaluation protocol currently used

- One fixed grouped development split is currently used for rapid iteration.
- Detector training labels tested include duplicate annotation records,
  maximum-completeness annotation selection and medoid annotation selection.
- Current best detectors and semantic branches were trained on the training
  portion only; the held-out 142 physical images have not yet been reused.
- This is appropriate for development, but the final leaderboard models should
  be retrained on all 707 physical images after architecture/hyperparameters are
  frozen.
- Five grouped folds are available/planned for paper-quality OOF evaluation,
  but the current result table is primarily one development fold.

## Frozen detector experiments

All comparable YOLOv8l experiments use one-class instance segmentation and the
same grouped development validation protocol.

| Run | Method | Best local mean PQ | Global PQ | TP | FP | FN | Public LB |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | YOLOv8l-seg, 1536, baseline | 0.35394 | not recorded | 843 | 519 | 750 | 0.33 |
| D2 | YOLOv8l-seg, 2048, standard augmentation | 0.37704 | 0.39365 | 991 | 763 | 602 | 0.32 |
| D3 | YOLOv8l-seg, 2048, conservative augmentation | 0.38815 | 0.40151 | 944 | 607 | 649 | 0.33 |
| D4 | YOLOv8l-seg, 2048, medoid labels | 0.37896 | 0.40209 | 926 | 563 | 667 | 0.33 |
| REF | YOLO11s-seg, 1024, four classes, max-completeness labels | 0.31318 | 0.32409 | 518 | 422 | 594 | not submitted |

Interpretation:

- D2 gained recall but generated too many false positives and scored worse on
  the public leaderboard.
- D3 is the primary detector; D4 is a more conservative complementary detector.
- The public leaderboard appears to reward precision/conservatism more strongly
  than the current local mean-PQ ranking alone suggests.

## Current two-branch method

### Instance branch

- Two YOLOv8l-seg models: D3 and D4.
- Test inference: `imgsz=2048`, base confidence 0.05, NMS IoU 0.70,
  `max_det=100`, retina masks.
- Predictions from D3/D4 are clustered by mask IoU. Duplicate candidates are
  merged and detector masks form instance seeds.

### Semantic/topology branch

Input channels are normalized grayscale, CLAHE/local contrast, and either a
radial-position map or Frangi vesselness. Models output three probability maps:
region, centerline and boundary.

| ID | Model | Training objective |
|---|---|---|
| S0 | U-Net, ResNet34 | region BCE + Dice |
| S1 | U-Net++, EfficientNet-B3 | region BCE + Dice |
| S2 | U-Net++, EfficientNet-B3 | region + centerline + Skeleton Recall Loss |
| S3 | U-Net++, EfficientNet-B3 | region + centerline + boundary + clDice |
| S4 | U-Net++, EfficientNet-B3, Frangi input | structural auxiliary losses; running/not yet reported |
| S5 | iterative residual U-Net++ | second residual expert + structural losses; running/not yet reported |

Training uses 768 tiles for S0-S4, 512 for S5, positive-biased crop sampling,
random rotations/flips and medoid annotation supervision. Inference uses 50%
overlapping tiles and Hann blending.

### Fusion

1. Cluster D3/D4 candidates by mask IoU.
2. Erode each detector mask to create a distinct marker/seed.
3. Form a semantic probability field, optionally boosted by the centerline map.
4. Apply seeded watershed inside semantic foreground.
5. Clip each result to an expanded detector neighborhood.
6. Keep the connected component touching its detector seed.
7. Fall back to detector mask if semantic refinement becomes too small.
8. Enforce zero overlap between final instances.
9. Optional rescue of elongated semantic components without detector seeds;
   this is disabled in current submitted runs.

## Fusion validation and leaderboard results

| Run | Semantic branch | det conf | sem threshold | min area | Mean PQ | Global PQ | SQ | RQ | TP | FP | FN | Public LB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P4-S0 | S0 U-Net | 0.25 | 0.55 | 200 | 0.39884 | 0.41101 | 0.61972 | 0.60158 | 1,039 | 726 | 554 | 0.34 |
| P4-S1 | S1 U-Net++ | 0.25 | 0.55 | 200 | **0.40776** | **0.41902** | **0.62463** | **0.61290** | **1,051** | 708 | **542** | 0.34 |
| P4-S2 | S2 skeleton recall | 0.30 | 0.55 | 200 | 0.40253 | 0.41811 | 0.61363 | 0.60551 | 1,010 | **622** | 583 | **0.35** |
| P4-S3 | S3 clDice + boundary | 0.25 | 0.35 | 200 | 0.36162 | 0.37646 | 0.59212 | 0.56035 | 973 | 800 | 620 | 0.31 |

Public leaderboard history:

| Submission | Public score |
|---|---:|
| YOLOv8l 1536 baseline | 0.33 |
| D2 detector | 0.32 |
| D3 detector | 0.33 |
| D4 detector | 0.33 |
| Fusion S0 | 0.34 |
| Fusion S1 | 0.34 |
| Fusion S2 | **0.35** |
| Fusion S3 | 0.31 |

Important observations:

1. Local ranking is directionally useful but not perfectly aligned with public
   ranking. S1 has the best local mean PQ, while the lower-FP S2 is best on the
   public leaderboard.
2. S1 and S2 have almost identical global PQ (0.4190 vs 0.4181), while mean
   per-annotation PQ prefers S1. The public test may resemble the global/precision
   criterion more than the current mean-over-annotators selection criterion.
3. S3 demonstrates that topology losses are not automatically beneficial.
   It is poorly calibrated, selects the lower edge semantic threshold 0.35,
   raises FP to 800 and degrades both local and public scores.
4. S0/S1/S2 choose semantic threshold 0.55, the upper edge of the first grid,
   and every run chooses `min_area=200`, also the upper edge. A wider conservative
   grid is not yet exhausted.
5. The public scores are shown with only two decimals. The exact improvement
   inside each displayed bin is unknown.
6. Validation/test semantic caches have now been separated into `val/` and
   `test/` namespaces to prevent filename collisions. Train/test filename
   intersection should still be audited before spending another submission.

## Public notebook audit

One reproducible public reference notebook trains both models internally:

- YOLO11s-seg, four chirality classes, 1024 input, 30 epochs.
- Chooses the annotation set with maximum instance count/area for each physical
  image.
- ResNet18 U-Net crop refiner, 256 crops, BCE + Dice, eight epochs.
- Detector proposals provide crop bounds; the detector seed is not a refiner
  input.
- No official local end-to-end PQ, no explicit non-overlap assignment and no
  hard-negative crop training.

Older public notebook titles claim scores around 0.55 or higher, but the metric
was changed and the leaderboard re-scored. Their present PQ score and exact
validity must be verified before treating them as achievable references.

## Current hypotheses

Please challenge these rather than accepting them blindly:

1. **Precision is currently more valuable than extra recall.** S2 beats S1 on
   public LB despite fewer TP/more FN locally because it removes 86 FP.
2. **The main remaining problem may be instance proposal quality, not semantic
   region segmentation.** All fusion systems still inherit detector misses and
   merge/split behavior.
3. **The current development fold/annotator averaging may be misaligned with
   hidden GT.** Model selection should use grouped OOF predictions and possibly
   a physical-image aggregation closer to the official hidden annotation policy.
4. **Training on only 80% of available physical images leaves legitimate
   leaderboard performance unused.** After hyperparameters are frozen, D3/D4
   and the chosen semantic branch should be retrained on all 707 images.
5. **Medoid hard labels may discard useful annotator information.** Random
   annotator sampling or soft pixel consensus/uncertainty weighting may improve
   robustness.
6. **A detection-free semantic connected-component approach is risky** because
   nearby filament structures merge, but a semantic branch could rescue only
   high-confidence, elongated candidates under strict gates.
7. **PQ-oriented training/selection is needed.** Dice/clDice improvements can
   reduce instance PQ if they connect neighboring filaments or enlarge masks.

## What I want in your response

Provide the following, with explicit reasoning tied to the evidence above:

1. **Feasibility assessment:** Is public PQ 0.70-0.80 realistic after the
   August PQ re-score? Find current legitimate post-rescore references if
   possible. Give realistic conservative/expected/stretch targets.
2. **Metric audit:** Identify any likely mismatch between the local evaluator,
   multi-annotator aggregation and Kaggle scoring. Specify tests needed before
   more training.
3. **Bottleneck diagnosis:** Use TP/FP/FN/SQ/RQ and local-vs-public behavior to
   rank missed detection, false positives, mask boundary quality, split/merge,
   annotator disagreement and domain shift.
4. **Prioritized experiment table:** At most 10 experiments, each with one main
   changed factor, expected signal/gain, GPU/runtime cost, dependencies, success
   threshold and stop rule. Separate cheap post-processing from retraining.
5. **Two-week four-GPU schedule:** What should run concurrently and what must
   wait for evidence? Reserve submission budget and avoid public-LB overfitting.
6. **Full-data final training protocol:** Explain exactly how to move from OOF
   development to training on all 707 images without losing calibrated
   thresholds. Include whether to ensemble grouped folds or retrain one final
   model initialized from the best checkpoints.
7. **Architecture decision:** Compare improving YOLO proposals, Mask2Former/
   Mask DINO/CondInst, U-Net instance separation, StarDist-like representations,
   boundary/centerline multitask learning and proposal-conditioned refinement.
   Recommend only methods justified by this dataset size and four-GPU budget.
8. **Multi-annotator strategy:** Compare duplicate labels, medoid, random
   annotator per epoch, soft consensus and uncertainty-aware loss. Distinguish
   leaderboard strategy from scientifically defensible paper evaluation.
9. **Fusion/ensemble plan:** Specify matching, mask voting, score calibration,
   overlap assignment, rescue gating and how to tune them using OOF caches.
10. **Error analysis:** Define plots/tables and an automated taxonomy that will
    determine the next experiment. Include size/thickness, disk position,
    observatory/year, annotator agreement, split/merge and FP morphology.
11. **Paper contribution:** Suggest a narrow, defensible contribution and an
    ablation table suitable for a six-page COMP3020 final report.
12. **Immediate next commands/config changes:** End with the exact next three
    actions you would take, not a broad wish list.

If information is missing, state what artifact or statistic is required rather
than inventing it. Clearly distinguish evidence, inference and speculation.
