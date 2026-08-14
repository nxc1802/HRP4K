from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .coco import load_split
from .identity import verify_dataset_identity
from .paths import SPLITS


def dataset_manifest(data_dir: Path, split: str) -> dict[str, Any]:
    """Build dataset provenance metadata for an inference experiment."""
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    annotation_path = data_dir / f"{split}.json"
    return {
        "root": str(data_dir.resolve()), "split": split,
        "benchmark_label": manifest.get("benchmark_label", "unverified"),
        "official_dataset_identity": manifest.get("official_dataset_identity"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None,
        "annotation_sha256": manifest.get("annotation_sha256", {}).get(split),
        "view_annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest() if annotation_path.is_file() else None,
    }
