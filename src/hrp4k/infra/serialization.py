from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, indent: int = 2) -> Path:
    """Write *payload* as JSON to *path*, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    return path


def write_text(path: Path, content: str) -> Path:
    """Write text to *path*, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
