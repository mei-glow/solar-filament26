import argparse
from pathlib import Path

from filament.config import Paths, resolve_checkpoint


def test_run_paths_are_scoped(tmp_path: Path):
    paths = Paths(tmp_path / "data.zip", tmp_path)
    assert paths.run_dir("a") != paths.run_dir("b")
    assert paths.experiment_results("a") == tmp_path / "results" / "a"


def test_checkpoint_resolution_is_explicit(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    paths = Paths(tmp_path / "data.zip", tmp_path)
    assert resolve_checkpoint(paths, "unused", checkpoint) == checkpoint.resolve()
