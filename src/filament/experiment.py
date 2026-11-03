from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def base_metadata() -> dict[str, Any]:
    try:
        import torch

        torch_info = {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
        }
    except Exception as error:  # pragma: no cover - diagnostic only
        torch_info = {"torch_error": repr(error)}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": sys.argv,
        "git_commit": git_commit(),
        **torch_info,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def record_event(artifact_root: Path, run_name: str, stage: str, payload: dict[str, Any]) -> Path:
    """Write one immutable registry event; safe for concurrent four-GPU runs."""
    event = {**base_metadata(), "run_name": run_name, "stage": stage, **payload}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = artifact_root / "registry" / f"{stamp}_{os.getpid()}_{run_name}_{stage}.json"
    write_json(path, event)
    return path
