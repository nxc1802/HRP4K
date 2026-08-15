from __future__ import annotations

import copy
from dataclasses import fields, asdict
from pathlib import Path
from typing import Any

from .loader import load_yaml
from .schema import (
    HRP4KConfig, ExperimentConfig, DatasetConfig, DetectorConfig,
    MethodConfig, RuntimeConfig, TrainingConfig, EvaluationConfig, OutputConfig,
)

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"

_SECTION_CLASSES = {
    "experiment": ExperimentConfig,
    "dataset": DatasetConfig,
    "detector": DetectorConfig,
    "method": MethodConfig,
    "runtime": RuntimeConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _build_section(cls, raw: dict[str, Any]):
    """Instantiate a dataclass from a dict, failing fast on unknown fields."""
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown fields in {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def _find_layer_file(dir_path: Path, name: str) -> Path | None:
    candidates = [
        dir_path / f"{name}.yaml",
        dir_path / f"{name.replace('-', '_')}.yaml",
        dir_path / f"{name.replace('_', '-')}.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_layer(path: Path, section_name: str | None = None) -> dict[str, Any]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        return {}
    if section_name and section_name not in data and not any(k in _SECTION_CLASSES for k in data):
        return {section_name: data}
    return data


def resolve(
    *,
    config_path: Path | None = None,
    detector: str | None = None,
    method: str | None = None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
    configs_dir: Path | None = None,
) -> HRP4KConfig:
    """Resolve modular YAML layers into a concrete :class:`HRP4KConfig`.

    Merge order (later wins):
        base.yaml → detector yaml → method yaml → profile yaml → experiment yaml → CLI overrides
    """
    root = Path(configs_dir or _CONFIGS_DIR)
    layers: list[dict[str, Any]] = []

    # Base defaults
    base_path = root / "base.yaml"
    if base_path.is_file():
        layers.append(_load_layer(base_path))

    # Detector layer
    if detector:
        det_file = _find_layer_file(root / "detectors", detector)
        if det_file:
            layers.append(_load_layer(det_file, "detector"))

    # Method layer
    if method:
        meth_file = _find_layer_file(root / "methods", method)
        if meth_file:
            layers.append(_load_layer(meth_file, "method"))

    # Profile layer
    if profile:
        prof_file = _find_layer_file(root / "profiles", profile)
        if prof_file:
            layers.append(_load_layer(prof_file))

    # Experiment file (full config)
    if config_path:
        layers.append(_load_layer(config_path))

    # CLI / programmatic overrides
    if overrides:
        layers.append(overrides)

    # Merge all layers
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer)

    # Build typed config
    sections: dict[str, Any] = {}
    for name, cls in _SECTION_CLASSES.items():
        raw = merged.get(name, {})
        if isinstance(raw, dict):
            sections[name] = _build_section(cls, raw)
        else:
            sections[name] = cls()

    return HRP4KConfig(
        schema_version=merged.get("schema_version", HRP4KConfig.schema_version),
        **sections,
    )


def to_dict(config: HRP4KConfig) -> dict[str, Any]:
    """Serialize a resolved config to a plain dict."""
    return asdict(config)
