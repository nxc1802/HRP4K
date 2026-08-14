from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; requires PyYAML from the vision dependencies."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Config loading requires PyYAML from the vision dependencies") from exc
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_config_layers(*paths: Path) -> list[dict[str, Any]]:
    """Load an ordered sequence of YAML config files."""
    return [load_yaml(p) for p in paths if p.is_file()]
