"""Runtime inspection and manifest verification for external adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def check_external_manifest(manifest: dict[str, Any], base_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Check whether configured external runner paths and resources exist."""
    errors: list[str] = []
    root = base_dir or Path.cwd()

    for key in ("python", "config", "checkpoint", "adapter"):
        if key in manifest and manifest[key]:
            val = manifest[key]
            path = Path(val) if Path(val).is_absolute() else root / val
            if not path.exists():
                errors.append(f"Missing external resource '{key}': {path}")

    return len(errors) == 0, errors
