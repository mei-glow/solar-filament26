# P1 audit: public YOLO + U-Net notebook

Reference: `ektarr/yolo-u-net-solar-filament-segmentation`, version 1.  
Pulled from the Kaggle kernels API on 2026-10-23 and stored unchanged as
`notebooks/reference/yolo-u-net-solar-filament-segmentation.ipynb`.

## Reproducibility and leakage check

- Kaggle metadata lists only the official `filament-segmentation-2026` competition input.
- It does not attach an external dataset, model dataset or another kernel as a data source.
- Both YOLO and the crop refiner are trained inside the notebook; no private `.pt`/`.pth` checkpoint is loaded.
- The train/validation split is made on unique physical filenames, then one annotation record is selected per filename. This avoids physical-image leakage between the notebook's train and validation sets.
- No test annotation path, external MAGFiLO label file or hand-authored prediction was found in the source.

This makes the notebook substantially more reproducible than the earlier HDJoJo inference notebook. Its displayed public score is still a public-leaderboard result, not evidence about the private leaderboard or cross-validation performance.

## Exact recipe found in source

Detector:

- YOLO11s-seg pretrained checkpoint.
- Input size 1024, batch 4, 30 epochs, patience 5.
- Four chirality categories (`Left`, `Right`, `Unidentifiable`, `Ambiguous`) during training; submission ignores category labels.
- `mask_ratio=2`, `overlap_mask=False`.
- `mosaic=0.3`, `degrees=10`, `scale=0.3`, `hsv_v=0.3`; flips disabled.
- For each physical image, it chooses the annotator with maximum `(instance_count, total_area)`.

Refiner:

- ResNet18 U-Net-style crop model.
- One grayscale channel repeated to RGB, crop size 256.
- GT instance crops with context 1.8, no negative crops.
- BCE + Dice, 8 epochs, validation threshold 0.65.
- At inference, each YOLO mask supplies only the crop bounds; the mask itself is not an input channel.

## Weaknesses that must not be copied blindly

- There is no official local PQ evaluation or PQ-based checkpoint selection.
- The annotator-selection rule favors more/larger labels; it is not a consensus medoid.
- The refiner has no hard negatives and no detector-seed input, so it can segment neighboring dark structures inside a proposal crop.
- Refined masks are exported independently without explicit duplicate removal or overlap assignment.
- Refiner validation uses crop Dice, which does not measure instance split/merge or end-to-end PQ.
- The source does not establish whether the gain comes from multi-class YOLO, `mask_ratio=2`, annotator selection, detector training, the refiner, or their combination.

## Reproduction/ablation mapping in this repository

- `prepare --annotation-strategy max-completeness --use-categories` reproduces its label selection and four-class targets.
- `train --model yolo11s-seg.pt --imgsz 1024 --mask-ratio 2 --no-overlap-mask --no-single-cls` reproduces its detector-specific settings.
- Our P3 refiner adds explicit hard negatives and a detector-seed channel. The public image-only recipe remains a reference ablation, not the default final model.
- Every detector/refiner variant must be evaluated with the same cached official PQ pipeline before promotion.
