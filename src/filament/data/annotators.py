from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


def _union_rle(image: dict, annotations: list[dict]) -> dict:
    rles = []
    for annotation in annotations:
        rles.append(mask_utils.merge(mask_utils.frPyObjects(annotation["segmentation"], image["height"], image["width"])))
    if not rles:
        return mask_utils.encode(np.asfortranarray(np.zeros((image["height"], image["width"]), dtype=np.uint8)))
    return mask_utils.merge(rles, intersect=False)


def medoid_image_ids(coco: dict) -> set[str]:
    images_by_id = {str(image["id"]): image for image in coco["images"]}
    anns_by_id: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_id[str(annotation["image_id"])].append(annotation)
    ids_by_file: dict[str, list[str]] = defaultdict(list)
    for image in coco["images"]:
        ids_by_file[image["file_name"]].append(str(image["id"]))

    selected: set[str] = set()
    for image_ids in ids_by_file.values():
        if len(image_ids) == 1:
            selected.add(image_ids[0])
            continue
        rles = [_union_rle(images_by_id[image_id], anns_by_id[image_id]) for image_id in image_ids]
        ious = mask_utils.iou(rles, rles, [0] * len(rles))
        selected.add(image_ids[int(np.argmax(ious.mean(axis=1)))])
    return selected


def select_annotation_records(manifest: pd.DataFrame, coco: dict, strategy: str, seed: int) -> pd.DataFrame:
    if strategy == "duplicate":
        return manifest.copy()
    if strategy == "first":
        return manifest.sort_values("image_id").groupby("file_name", as_index=False).head(1).copy()
    if strategy == "medoid":
        selected = medoid_image_ids(coco)
        return manifest[manifest["image_id"].astype(str).isin(selected)].copy()
    if strategy == "max-completeness":
        anns_by_id: dict[str, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            anns_by_id[str(annotation["image_id"])].append(annotation)
        scores = {
            image_id: (len(annotations), sum(float(item.get("area", 0.0)) for item in annotations))
            for image_id, annotations in anns_by_id.items()
        }
        chosen = []
        for _, group in manifest.groupby("file_name"):
            chosen.append(max(group["image_id"].astype(str), key=lambda image_id: scores.get(image_id, (0, 0.0))))
        return manifest[manifest["image_id"].astype(str).isin(chosen)].copy()
    if strategy == "random-single":
        return (
            manifest.groupby("file_name", group_keys=False)
            .sample(n=1, random_state=seed)
            .sort_values(["file_name", "image_id"])
            .copy()
        )
    raise ValueError(f"Unknown annotation strategy: {strategy}")
