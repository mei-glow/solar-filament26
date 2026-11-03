import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md("""# Solar filament segmentation — reproducible server baseline

This notebook trains a one-class YOLO instance-segmentation baseline, tunes its
post-processing with leakage-safe grouped validation, evaluates Panoptic
Quality (PQ), and produces a Kaggle submission.

All generated data, checkpoints, logs, predictions, metrics and submissions
are written below `/workspace/solar-filament-artifacts`. The original competition zip and
this notebook remain under `/workspace/solar-filament26`.

The physical image filename is the split group because the annotation JSON may
contain multiple independent annotation sets for the same JPEG. This prevents
the same pixels from appearing in both train and validation.
"""),
    code("""# Run once in the server environment.
%pip install -q \"ultralytics>=8.3,<9\" \"pycocotools>=2.0.7\" \"scikit-learn>=1.4\" \"scipy>=1.11\" pyyaml tqdm"""),
    code("""from __future__ import annotations

import csv
import gc
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
from ultralytics import YOLO


@dataclass(frozen=True)
class Config:
    NOTEBOOK_ROOT: Path = Path(\"/workspace/solar-filament26/notebooks\")
    ZIP_PATH: Path = Path(\"/workspace/solar-filament26/filament-segmentation-2026.zip\")
    ARTIFACT_ROOT: Path = Path(\"/workspace/solar-filament-artifacts\")
    EXTRACT_ROOT: Path = ARTIFACT_ROOT / \"data_extracted\"
    YOLO_ROOT: Path = ARTIFACT_ROOT / \"yolo_dataset\"
    RUNS_ROOT: Path = ARTIFACT_ROOT / \"runs\"
    RESULTS_ROOT: Path = ARTIFACT_ROOT / \"results\"
    MODEL: str = \"yolov8l-seg.pt\"
    RUN_NAME: str = \"yolov8l_seg_grouped_baseline\"
    IMG_SIZE: int = 1536       # Set 2048 after the pipeline is verified.
    EPOCHS: int = 60
    BATCH: int = 2             # Global batch; reduce to 1 if CUDA OOM.
    WORKERS: int = 8
    VAL_FRACTION: float = 0.20
    SEED: int = 3020
    BASE_CONF: float = 0.05    # Cache broad predictions, then tune threshold.
    NMS_IOU: float = 0.70
    MAX_DET: int = 100
    SAVE_PERIOD: int = 1       # Keep every epoch checkpoint.
    TRAIN: bool = True
    RESUME: bool = False


CFG = Config()
for path in (CFG.ARTIFACT_ROOT, CFG.EXTRACT_ROOT, CFG.YOLO_ROOT, CFG.RUNS_ROOT, CFG.RESULTS_ROOT):
    path.mkdir(parents=True, exist_ok=True)

assert CFG.ZIP_PATH.is_file(), f\"Missing dataset zip: {CFG.ZIP_PATH}\"
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
DEVICE = \"0,1\" if torch.cuda.device_count() >= 2 else (0 if torch.cuda.is_available() else \"cpu\")
print(json.dumps({**{k: str(v) for k, v in asdict(CFG).items()}, \"DEVICE\": DEVICE}, indent=2))
print(\"GPUs:\", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"""),
    code("""# Extract once. zipfile also handles a top-level competition directory.
sentinel = CFG.EXTRACT_ROOT / \".complete\"
if not sentinel.exists():
    print(f\"Extracting {CFG.ZIP_PATH} -> {CFG.EXTRACT_ROOT}\")
    with zipfile.ZipFile(CFG.ZIP_PATH) as zf:
        zf.extractall(CFG.EXTRACT_ROOT)
    sentinel.write_text(\"ok\\n\")

json_candidates = list(CFG.EXTRACT_ROOT.rglob(\"MAGFiLO_1.0_Annotations_kaggle2026_train.json\"))
assert len(json_candidates) == 1, f\"Expected one annotation JSON, found {json_candidates}\"
ANN_PATH = json_candidates[0]
DATA_ROOT = ANN_PATH.parent.parent
TRAIN_IMAGE_DIR = DATA_ROOT / \"train\" / \"train_images\"
TEST_IMAGE_DIR = DATA_ROOT / \"test\" / \"test_images\"
assert TRAIN_IMAGE_DIR.is_dir() and TEST_IMAGE_DIR.is_dir()
print(DATA_ROOT)
print(\"train JPEGs:\", len(list(TRAIN_IMAGE_DIR.glob(\"*.jpeg\"))))
print(\"test JPEGs :\", len(list(TEST_IMAGE_DIR.glob(\"*.jpeg\"))))"""),
    code("""# Convert each independent annotation set to a unique YOLO sample.
# Repeated physical images are hard-linked where possible to avoid wasting disk.
with ANN_PATH.open() as f:
    coco = json.load(f)

images = pd.DataFrame(coco[\"images\"])
anns_by_image = defaultdict(list)
for ann in coco[\"annotations\"]:
    anns_by_image[ann[\"image_id\"]].append(ann)

splitter = GroupShuffleSplit(n_splits=1, test_size=CFG.VAL_FRACTION, random_state=CFG.SEED)
train_idx, val_idx = next(splitter.split(images, groups=images[\"file_name\"]))
images[\"split\"] = \"train\"
images.loc[val_idx, \"split\"] = \"val\"
assert not (set(images.loc[train_idx, \"file_name\"]) & set(images.loc[val_idx, \"file_name\"]))

def safe_stem(image_id: str) -> str:
    return image_id.replace(\"/\", \"_\").replace(\"\\\\\", \"_\")

def materialize(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

for split in (\"train\", \"val\"):
    (CFG.YOLO_ROOT / \"images\" / split).mkdir(parents=True, exist_ok=True)
    (CFG.YOLO_ROOT / \"labels\" / split).mkdir(parents=True, exist_ok=True)

manifest_rows = []
for row in tqdm(images.itertuples(index=False), total=len(images), desc=\"COCO -> YOLO\"):
    stem = safe_stem(row.id)
    src = TRAIN_IMAGE_DIR / row.file_name
    dst = CFG.YOLO_ROOT / \"images\" / row.split / f\"{stem}.jpeg\"
    label_path = CFG.YOLO_ROOT / \"labels\" / row.split / f\"{stem}.txt\"
    assert src.is_file(), src
    materialize(src, dst)

    lines = []
    for ann in anns_by_image[row.id]:
        for polygon in ann[\"segmentation\"]:
            pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            if len(pts) < 3:
                continue
            pts[:, 0] = np.clip(pts[:, 0] / row.width, 0, 1)
            pts[:, 1] = np.clip(pts[:, 1] / row.height, 0, 1)
            coords = \" \".join(f\"{x:.7f} {y:.7f}\" for x, y in pts)
            lines.append(f\"0 {coords}\")
    label_path.write_text(\"\\n\".join(lines) + (\"\\n\" if lines else \"\"))
    manifest_rows.append({\"sample_stem\": stem, \"image_id\": row.id, \"file_name\": row.file_name, \"split\": row.split})

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(CFG.RESULTS_ROOT / \"split_manifest.csv\", index=False)

dataset_yaml = {
    \"path\": str(CFG.YOLO_ROOT),
    \"train\": \"images/train\",
    \"val\": \"images/val\",
    \"names\": {0: \"filament\"},
}
DATA_YAML = CFG.YOLO_ROOT / \"dataset.yaml\"
DATA_YAML.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))
print(manifest.groupby(\"split\").agg(records=(\"image_id\", \"size\"), physical_images=(\"file_name\", \"nunique\")))
print(DATA_YAML.read_text())"""),
    code("""# Train. Every epoch checkpoint, results.csv and plots stay under ARTIFACT_ROOT.
run_dir = CFG.RUNS_ROOT / CFG.RUN_NAME
last_ckpt = run_dir / \"weights\" / \"last.pt\"
best_ckpt = run_dir / \"weights\" / \"best.pt\"

if CFG.TRAIN:
    if CFG.RESUME and last_ckpt.exists():
        model = YOLO(str(last_ckpt))
        train_result = model.train(resume=True)
    else:
        model = YOLO(CFG.MODEL)
        train_result = model.train(
            data=str(DATA_YAML),
            project=str(CFG.RUNS_ROOT),
            name=CFG.RUN_NAME,
            exist_ok=True,
            imgsz=CFG.IMG_SIZE,
            epochs=CFG.EPOCHS,
            batch=CFG.BATCH,
            workers=CFG.WORKERS,
            device=DEVICE,
            seed=CFG.SEED,
            deterministic=True,
            single_cls=True,
            cache=False,
            amp=True,
            patience=20,
            save=True,
            save_period=CFG.SAVE_PERIOD,
            plots=True,
            close_mosaic=10,
        )

assert best_ckpt.is_file(), f\"Missing trained checkpoint: {best_ckpt}\"
print(\"Best checkpoint:\", best_ckpt)"""),
    code("""# Local PQ evaluator. Predictions and GT are scored per annotation set.
def polygon_to_mask(polygons, height=2048, width=2048):
    rles = mask_utils.frPyObjects(polygons, height, width)
    merged = mask_utils.merge(rles)
    return mask_utils.decode(merged).astype(bool)

def pairwise_iou(gt_masks, pred_masks):
    if not gt_masks or not pred_masks:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    gt = np.stack(gt_masks).reshape(len(gt_masks), -1)
    pr = np.stack(pred_masks).reshape(len(pred_masks), -1)
    inter = gt.astype(np.uint32) @ pr.astype(np.uint32).T
    union = gt.sum(1)[:, None] + pr.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)

def pq_one(gt_masks, pred_masks, threshold=0.5):
    ious = pairwise_iou(gt_masks, pred_masks)
    if ious.size:
        rows, cols = linear_sum_assignment(-ious)
        matched = ious[rows, cols] > threshold
        tp = int(matched.sum())
        iou_sum = float(ious[rows[matched], cols[matched]].sum())
    else:
        tp, iou_sum = 0, 0.0
    fp, fn = len(pred_masks) - tp, len(gt_masks) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return (iou_sum / denom if denom else 1.0), tp, fp, fn

def resolve_instances(masks, scores, conf=0.30, min_area=25, duplicate_iou=0.80):
    keep = np.flatnonzero(scores >= conf)
    keep = keep[np.argsort(scores[keep])[::-1]]
    accepted, occupied = [], None
    for idx in keep:
        mask = masks[idx].astype(bool)
        if mask.sum() < min_area:
            continue
        duplicate = False
        for old in accepted:
            inter = np.logical_and(mask, old).sum()
            union = np.logical_or(mask, old).sum()
            if union and inter / union >= duplicate_iou:
                duplicate = True
                break
        if duplicate:
            continue
        # PQ expects non-overlapping panoptic instances. Higher-confidence masks win.
        if occupied is not None:
            mask = np.logical_and(mask, ~occupied)
        if mask.sum() < min_area:
            continue
        accepted.append(mask)
        occupied = mask.copy() if occupied is None else np.logical_or(occupied, mask)
    return accepted

gt_by_id = {
    row.id: [polygon_to_mask(a[\"segmentation\"], row.height, row.width) for a in anns_by_image[row.id]]
    for row in images[images[\"split\"] == \"val\"].itertuples(index=False)
}"""),
    code("""# Cache broad validation predictions once, then tune only post-processing.
model = YOLO(str(best_ckpt))
val_physical = sorted(images.loc[images[\"split\"] == \"val\", \"file_name\"].unique())
raw_val = {}
for file_name in tqdm(val_physical, desc=\"Validation inference\"):
    result = model.predict(
        str(TRAIN_IMAGE_DIR / file_name), imgsz=CFG.IMG_SIZE,
        conf=CFG.BASE_CONF, iou=CFG.NMS_IOU, max_det=CFG.MAX_DET,
        retina_masks=True, verbose=False,
    )[0]
    if result.masks is None:
        raw_val[file_name] = (np.empty((0, 2048, 2048), bool), np.empty(0, np.float32))
    else:
        raw_val[file_name] = (
            result.masks.data.cpu().numpy().astype(bool),
            result.boxes.conf.cpu().numpy(),  # confidence, not class id
        )

grid_rows = []
for conf in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    for min_area in (25, 50, 100, 200):
        scores = []
        totals = np.zeros(3, dtype=np.int64)
        for row in images[images[\"split\"] == \"val\"].itertuples(index=False):
            masks, confidences = raw_val[row.file_name]
            pred = resolve_instances(masks, confidences, conf=conf, min_area=min_area)
            pq, tp, fp, fn = pq_one(gt_by_id[row.id], pred)
            scores.append(pq)
            totals += (tp, fp, fn)
        grid_rows.append({\"conf\": conf, \"min_area\": min_area, \"mean_pq\": np.mean(scores),
                          \"tp\": totals[0], \"fp\": totals[1], \"fn\": totals[2]})

tuning = pd.DataFrame(grid_rows).sort_values(\"mean_pq\", ascending=False)
tuning.to_csv(CFG.RESULTS_ROOT / \"postprocess_tuning.csv\", index=False)
display(tuning.head(12))
BEST_CONF = float(tuning.iloc[0].conf)
BEST_MIN_AREA = int(tuning.iloc[0].min_area)
print(\"Selected:\", BEST_CONF, BEST_MIN_AREA)"""),
    code("""# Full-resolution test inference and Kaggle submission.
def mask_to_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle[\"counts\"].decode(\"utf-8\")

submission_rows = []
per_image_rows = []
for img_path in tqdm(sorted(TEST_IMAGE_DIR.glob(\"*.jpeg\")), desc=\"Test inference\"):
    result = model.predict(
        str(img_path), imgsz=CFG.IMG_SIZE,
        conf=CFG.BASE_CONF, iou=CFG.NMS_IOU, max_det=CFG.MAX_DET,
        retina_masks=True, verbose=False,
    )[0]
    if result.masks is None:
        pred_masks = []
    else:
        masks = result.masks.data.cpu().numpy().astype(bool)
        confidences = result.boxes.conf.cpu().numpy()
        pred_masks = resolve_instances(masks, confidences, conf=BEST_CONF, min_area=BEST_MIN_AREA)

    for i, mask in enumerate(pred_masks, start=1):
        submission_rows.append({\"filament_id\": f\"{img_path.stem}_{i}\", \"segmentation_rle\": mask_to_rle(mask)})
    per_image_rows.append({\"file_name\": img_path.name, \"instances\": len(pred_masks)})

submission_path = CFG.RESULTS_ROOT / f\"submission_{CFG.RUN_NAME}.csv\"
pd.DataFrame(submission_rows, columns=[\"filament_id\", \"segmentation_rle\"]).to_csv(submission_path, index=False, quoting=csv.QUOTE_MINIMAL)
pd.DataFrame(per_image_rows).to_csv(CFG.RESULTS_ROOT / \"test_instance_counts.csv\", index=False)
print(\"Submission:\", submission_path)
print(\"Rows:\", len(submission_rows), \"images with predictions:\", sum(r[\"instances\"] > 0 for r in per_image_rows))
display(pd.DataFrame(per_image_rows)[\"instances\"].describe())"""),
    md("""## Next experiments for the paper

Keep this run unchanged as the principal baseline. Planned ablations should
change one factor at a time: image size, backbone size, normalization/CLAHE,
multi-annotator sampling, confidence calibration, overlap assignment, and a
topology-aware refiner. Report grouped-validation PQ with TP/FP/FN, Kaggle PQ,
runtime, memory, and qualitative failure cases.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(r"notebooks/working/solar-filament-seg-improved.ipynb")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(target)
