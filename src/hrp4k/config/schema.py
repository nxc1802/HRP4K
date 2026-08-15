from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = "hrp4k.config.v1"


@dataclass
class DatasetConfig:
    root: str = "HRP4K"
    split: str = "test"
    release_id: str | None = None
    expected_hash: dict[str, str] | None = None


@dataclass
class DetectorConfig:
    name: str = "ultralytics"
    framework: str = "ultralytics"
    checkpoint: str = "yolo11m.pt"
    checkpoint_hash: str | None = None
    input_size: int = 640
    confidence: float = 0.05
    device: str | None = None


@dataclass
class MethodConfig:
    name: str = "resize"
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def tile_size(self) -> int:
        return int(self.parameters.get("tile_size", self.parameters.get("slice_width", 960)))

    @property
    def overlap(self) -> float:
        return float(self.parameters.get("overlap", 0.2))


@dataclass
class RuntimeConfig:
    device: str | None = None
    precision: str = "fp32"
    warmup: int = 20
    warmup_images: int | None = None
    batch_size: int = 1
    deterministic: bool = True
    limit: int | None = None

    def __post_init__(self):
        if self.warmup_images is not None:
            self.warmup = self.warmup_images


@dataclass
class TrainingConfig:
    epochs: int = 150
    image_size: int = 640
    batch: int = 16
    smoke: bool = False
    allow_full: bool = False
    seed: int = 42


@dataclass
class EvaluationConfig:
    coco: bool = True
    fppi: bool = True
    scale_metrics: bool = True
    confidence_threshold: float = 0.25
    iou_thresholds: list[float] = field(default_factory=lambda: [])


@dataclass
class OutputConfig:
    root: str = "outputs"
    study_id: str | None = None
    predictions: str | None = None


@dataclass
class ExperimentConfig:
    study_id: str | None = None
    experiment_name: str | None = None
    name: str | None = None
    seed: int = 42
    profile: str = "smoke"

    def __post_init__(self):
        if self.experiment_name is None and self.name is not None:
            self.experiment_name = self.name
        elif self.name is None and self.experiment_name is not None:
            self.name = self.experiment_name


@dataclass
class HRP4KConfig:
    """Canonical resolved experiment configuration."""

    schema_version: str = SCHEMA_VERSION
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
