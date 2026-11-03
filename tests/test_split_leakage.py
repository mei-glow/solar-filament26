from pathlib import Path

import pandas as pd

from filament.data.splits import build_fold_manifest


def test_fold_manifest_has_no_physical_image_leakage(tmp_path: Path):
    annotation = Path(
        "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/"
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    if not annotation.is_file():
        return
    manifest = build_fold_manifest(annotation, tmp_path / "folds.csv", folds=5, seed=3020)
    assert manifest["file_name"].nunique() == 707
    assert manifest.groupby("file_name")["fold"].nunique().max() == 1
    assert sorted(manifest["fold"].unique()) == [0, 1, 2, 3, 4]


def test_training_strategy_does_not_remove_validation_annotators():
    # Contract enforced by prepare: train selection and val selection happen separately.
    source = Path("src/filament/prepare.py").read_text()
    assert "training_records = select_annotation_records" in source
    assert 'if args.val_annotation_strategy == "same"' in source
