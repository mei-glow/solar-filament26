# MAGFiLO v1.0 — Partial Baseline Results

Snapshot: 2026-11-26

**20/25 completed target runs.**

Completed: U-Net++ + watershed, Mask R-CNN, Cascade Mask R-CNN, HTC (5 seeds each, 3020–3024).

SOLOv2: 0/5 valid results. Seed 3020 trained but its mask predictions collapsed; the remaining seeds are blocked.

See GitHub issue #1 for diagnosis.

Metric: test global PQ. Summary standard deviation uses ddof=1.

Individual test and available validation result JSONs are preserved in `individual/`.
