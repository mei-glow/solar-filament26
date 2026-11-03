# Experiment log

Machine-readable events are written independently to
`${ARTIFACT_ROOT}/registry/*.json`. This file records externally
observed results and promotion decisions.

| ID | Model/protocol | Local PQ | TP | FP | FN | Kaggle | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| E0 | YOLOv8l-seg, 1536, legacy grouped validation, conf 0.20, min area 200 | 0.353937 | 843 | 519 | 750 | 0.33 | Frozen baseline |
| P2-D2 | YOLOv8l-seg, 2048, standard augmentation, conf 0.20, min area 200 | 0.377036 | 991 | 763 | 602 | 0.32 | Reject: recall gain accompanied by excessive FP and lower LB |
| P2-D3 | YOLOv8l-seg, 2048, conservative augmentation, conf 0.25, min area 200 | 0.388150 | 944 | 607 | 649 | 0.33 | Promote as primary P4 detector |
| P2-D4 | YOLOv8l-seg, 2048, medoid training labels, conf 0.35, min area 200 | 0.378959 | 926 | 563 | 667 | 0.33 | Retain as secondary/ensemble detector |
| P2-REF | YOLO11s-seg, 1024, four-class max-completeness reproduction, conf 0.20, min area 200 | 0.313183 | 518 | 422 | 594 | — | Hold for reference-refiner fusion; validation protocol is not directly comparable |
| P4-S0 | D3+D4 seeded-watershed fusion with U-Net/ResNet34 semantic field, det conf 0.25, semantic threshold 0.55, min area 200 | 0.398837 | 1039 | 726 | 554 | 0.34 | Retain as region-only fusion baseline |
| P4-S1 | D3+D4 fusion with U-Net++/EfficientNet-B3 semantic field, det conf 0.25, semantic threshold 0.55, min area 200 | 0.407759 | 1051 | 708 | 542 | 0.34 | Best local mean PQ; retain for OOF/full-data study |
| P4-S2 | D3+D4 fusion with centerline + skeleton-recall branch, det conf 0.30, semantic threshold 0.55, min area 200 | 0.402531 | 1010 | 622 | 583 | 0.35 | Best public score; promote as precision-oriented primary |
| P4-S3 | D3+D4 fusion with boundary + clDice branch, det conf 0.25, semantic threshold 0.35, min area 200 | 0.361622 | 973 | 800 | 620 | 0.31 | Reject current loss/calibration; topology increased FP and hurt PQ |

P2 leaderboard screenshot received 2026-10-27. Kaggle displayed scores to two decimal places, so D3/D4 tying the baseline at 0.33 does not establish exact equality. The detector-only P2-REF run is not the complete public YOLO+U-Net pipeline and therefore is not expected to reproduce its reported score without the refiner/fusion stage.

P4 leaderboard results received 2026-10-29. S2 outperformed S1 publicly despite
lower local mean PQ; their global PQ values are nearly tied (0.41811 vs 0.41902)
and S2 has 86 fewer validation FP. This is evidence to prioritize conservative
precision, grouped OOF calibration and full-data retraining rather than choosing
only by one-fold mean PQ. Scores are displayed to two decimals.

## Required fields for new entries

- date and git commit;
- data/split hash and annotation strategy;
- model/initial checkpoint/config/seed;
- GPU and runtime;
- checkpoint path;
- PQ/SQ/RQ/TP/FP/FN and size-stratified metrics;
- post-processing parameters;
- Kaggle score if submitted;
- reject/promote decision and reason.
