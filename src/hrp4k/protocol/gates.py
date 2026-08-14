from __future__ import annotations

import hashlib
import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..data.paths import SPLITS
from ..data.audit import analyze_dataset
from ..data.identity import verify_dataset_identity


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def preflight(
    data_dir: Path, *, output_dir: Path | None = None, weights: Path | None = None,
    device: str | None = None, require_official: bool = False,
) -> dict[str, Any]:
    """Run read-only launch checks, except for a small temporary analysis report."""
    errors: list[str] = []
    warnings: list[str] = []
    hashes: dict[str, str] = {}
    for split in SPLITS:
        annotation = data_dir / f"{split}.json"
        image_dir = data_dir / split / "images"
        if not annotation.is_file():
            errors.append(f"missing annotation: {annotation}")
        else:
            hashes[split] = hashlib.sha256(annotation.read_bytes()).hexdigest()
        if not image_dir.is_dir():
            errors.append(f"missing image directory: {image_dir}")
    identity = verify_dataset_identity({"annotation_sha256": hashes})
    integrity: dict[str, Any] = {}
    if not errors:
        with tempfile.TemporaryDirectory() as directory:
            integrity = analyze_dataset(data_dir, Path(directory), quality_samples=0)["integrity"]
        errors.extend(integrity.get("errors", []))
        warnings.extend(integrity.get("warnings", []))
    if require_official and not identity["official_dataset_identity"]:
        errors.append("official run requested but annotation identity verification failed")
    if weights is not None and not weights.is_file():
        errors.append(f"missing weights: {weights}")
    if output_dir is not None:
        parent = output_dir if output_dir.exists() else output_dir.parent
        if not parent.exists():
            warnings.append(f"output parent will be created: {parent}")
    cuda = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        pass
    if device and str(device).startswith("cuda") and not cuda:
        errors.append(f"CUDA device requested but unavailable: {device}")
    disk_free_gb = shutil.disk_usage(data_dir).free / (1024 ** 3) if data_dir.exists() else None
    if disk_free_gb is not None and disk_free_gb < 5:
        warnings.append(f"low disk space: {disk_free_gb:.1f} GiB free")
    return {
        "status": "pass" if not errors else "fail", "dataset": "pass" if not errors else "fail",
        **identity, "cuda": cuda, "pycocotools": _available("pycocotools"),
        "ultralytics": _available("ultralytics"), "sahi": _available("sahi"),
        "rtdetr": _available("src") or _available("rtdetr"), "dfine": _available("dfine"),
        "disk_free_gb": disk_free_gb, "integrity": integrity, "errors": errors, "warnings": warnings,
    }
