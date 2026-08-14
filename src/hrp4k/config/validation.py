from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import HRP4KConfig


def validate(config: HRP4KConfig) -> list[str]:
    """Validate a resolved config before GPU initialization.

    Returns a list of error messages; empty means valid.
    """
    errors: list[str] = []

    # Dataset
    data_root = Path(config.dataset.root)
    if not data_root.is_dir():
        errors.append(f"dataset root does not exist: {data_root}")

    # Detector
    if not config.detector.name:
        errors.append("detector name is required")
    if config.detector.input_size <= 0:
        errors.append(f"detector input_size must be positive: {config.detector.input_size}")
    if not (0.0 < config.detector.confidence <= 1.0):
        errors.append(f"detector confidence must be in (0, 1]: {config.detector.confidence}")

    # Method
    if not config.method.name:
        errors.append("method name is required")

    # Runtime
    if config.runtime.precision not in ("fp32", "fp16"):
        errors.append(f"runtime precision must be fp32 or fp16: {config.runtime.precision}")
    if config.runtime.warmup < 0:
        errors.append(f"runtime warmup must be non-negative: {config.runtime.warmup}")

    # Training
    if config.training.epochs <= 0:
        errors.append(f"training epochs must be positive: {config.training.epochs}")
    if config.training.image_size <= 0:
        errors.append(f"training image_size must be positive: {config.training.image_size}")

    # Evaluation
    if not (0.0 < config.evaluation.confidence_threshold <= 1.0):
        errors.append(f"evaluation confidence_threshold must be in (0, 1]: {config.evaluation.confidence_threshold}")

    return errors
