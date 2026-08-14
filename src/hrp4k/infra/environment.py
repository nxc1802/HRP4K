from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def environment_snapshot() -> dict[str, Any]:
    """Collect runtime environment metadata for provenance tracking."""
    snapshot: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        snapshot.update({
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        snapshot.update({"torch": None, "cuda_version": None, "cuda_available": False, "gpu": None})
    try:
        import ultralytics
        snapshot["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    try:
        snapshot["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        snapshot["git_commit"] = None
    try:
        snapshot["pip_freeze"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (subprocess.SubprocessError, FileNotFoundError):
        snapshot["pip_freeze"] = []
    return snapshot


def runtime_metadata(
    device: str | None, precision: str, image_size: int, warmup: int,
) -> dict[str, Any]:
    """Collect runtime metadata for a specific inference run."""
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "device": device or "auto",
        "precision": precision,
        "batch_size": 1,
        "image_size": image_size,
        "warmup_images": warmup,
    }
    try:
        import torch
        metadata.update({
            "pytorch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        metadata.update({"pytorch": None, "cuda_version": None, "cuda_available": False, "gpu": None})
    try:
        metadata["commit_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        metadata["commit_sha"] = None
    return metadata
