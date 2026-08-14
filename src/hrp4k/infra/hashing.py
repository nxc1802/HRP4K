from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Deterministic JSON string for hashing purposes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_dict(value: Any) -> str:
    """SHA-256 hex digest of the canonical JSON representation of *value*."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's byte content."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment_id(config: dict[str, Any]) -> str:
    """Short deterministic identifier derived from the resolved experiment config."""
    return sha256_dict(config)[:12]
