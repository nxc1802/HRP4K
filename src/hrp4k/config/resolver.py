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
    """Instantiate a dataclass from a dict, ignoring unknown fields."""
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in valid})


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
        layers.append(load_yaml(base_path))

    # Detector layer
    if detector:
        detector_path = root / "detectors" / f"{detector}.yaml"
        if detector_path.is_file():
            layers.append(load_yaml(detector_path))

    # Method layer
    if method:
        method_path = root / "methods" / f"{method}.yaml"
        if method_path.is_file():
            layers.append(load_yaml(method_path))

    # Profile layer
    if profile:
        profile_path = root / "profiles" / f"{profile}.yaml"
        if profile_path.is_file():
            layers.append(load_yaml(profile_path))

    # Experiment file (full config)
    if config_path:
        layers.append(load_yaml(config_path))

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
