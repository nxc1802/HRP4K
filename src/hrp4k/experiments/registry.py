"""Experiment Registry — deterministic experiment matrix and config resolution.

Every official experiment has a canonical name and a frozen configuration.
The experiment ID is computed as SHA256(canonical_config) so that any change
in research-relevant parameters produces a new identity.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Any

from ..infra.hashing import experiment_id


# ---------------------------------------------------------------------------
# Experiment configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Fully-resolved, frozen experiment configuration."""
    name: str
    phase: str                          # "resolution" | "slicing" | "proposed"
    detector: str                       # "yolo11m" | "rtdetr-l"
    weights: str                        # pretrained checkpoint name
    resolution: str                     # "4k" | "2k" | "1k" | "640"
    imgsz: int | tuple[int, int] = 640  # actual pixel size
    batch: int = 16                     # physical batch
    accumulation: int = 1               # gradient accumulation steps
    effective_batch: int = 16
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_bias_lr: float = 0.1
    warmup_momentum: float = 0.8
    momentum: float = 0.937
    amp: bool = True
    rect: bool = True
    epochs: int = 150
    patience: int = 10
    seed: int = 42
    confidence: float = 0.001
    # Slicing-specific
    method: str = "resize"              # only used for slicing phase
    tile_size: int = 960
    overlap: float = 0.2
    frozen_checkpoint: str | None = None  # for slicing: path to frozen 640 checkpoint
    # Metadata
    dataset: str = "HRP4K"
    experiment_id: str = ""

    def __post_init__(self):
        self.effective_batch = self.batch * self.accumulation
        if not self.experiment_id:
            self.experiment_id = self._compute_id()

    def _compute_id(self) -> str:
        """Compute deterministic experiment ID from research-relevant fields."""
        canonical = {
            "detector": self.detector,
            "phase": self.phase,
            "resolution": self.resolution,
            "imgsz": self.imgsz,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "effective_batch": self.effective_batch,
            "amp": self.amp,
            "rect": self.rect,
            "seed": self.seed,
            "method": self.method,
            "dataset": self.dataset,
        }
        return experiment_id(canonical)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# YOLO11m base config
# ---------------------------------------------------------------------------

_YOLO11M_BASE = dict(
    detector="yolo11m",
    weights="yolo11m.pt",
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_bias_lr=0.1,
    momentum=0.937,
    amp=True,
    rect=True,
    epochs=150,
    patience=10,
    seed=42,
    confidence=0.001,
)

# ---------------------------------------------------------------------------
# RT-DETR-L base config
# ---------------------------------------------------------------------------

_RTDETRL_BASE = dict(
    detector="rtdetr-l",
    weights="rtdetr-l.pt",
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.01,
    weight_decay=0.0001,
    warmup_bias_lr=0.0,
    momentum=0.937,
    amp=True,      # overridden to False for 4K
    rect=True,
    epochs=150,
    patience=10,
    seed=42,
    confidence=0.001,
)

# ---------------------------------------------------------------------------
# Resolution matrix
# ---------------------------------------------------------------------------

_RESOLUTION_MAP = {
    "4k": {"imgsz": 3840, "batch": 2, "accumulation": 8},
    "2k": {"imgsz": 1920, "batch": 4, "accumulation": 4},
    "1k": {"imgsz": 960,  "batch": 16, "accumulation": 1},
    "640": {"imgsz": 640,  "batch": 16, "accumulation": 1},
}

# ---------------------------------------------------------------------------
# Build experiment matrix
# ---------------------------------------------------------------------------

def _build_resolution_experiments() -> dict[str, ExperimentConfig]:
    experiments = {}

    for res_name, res_cfg in _RESOLUTION_MAP.items():
        # YOLO11m
        yolo_name = f"yolo11m-resolution-{res_name}"
        experiments[yolo_name] = ExperimentConfig(
            name=yolo_name,
            phase="resolution",
            resolution=res_name,
            **{**_YOLO11M_BASE, **res_cfg},
        )

        # RT-DETR-L
        rtdetr_name = f"rtdetr-l-resolution-{res_name}"
        amp = False if res_name == "4k" else True  # FP32 for 4K Transformer
        experiments[rtdetr_name] = ExperimentConfig(
            name=rtdetr_name,
            phase="resolution",
            resolution=res_name,
            amp=amp,
            **{k: v for k, v in {**_RTDETRL_BASE, **res_cfg}.items() if k != "amp"},
        )

    return experiments


def _build_slicing_experiments() -> dict[str, ExperimentConfig]:
    experiments = {}
    methods = ["resize", "sliced-nms", "sahi", "perspective-grid"]

    for method in methods:
        method_slug = method  # e.g. "sliced-nms", "sahi"
        display = "full" if method == "resize" else method_slug

        # YOLO11m slicing
        yolo_name = f"yolo11m-slicing-{display}"
        experiments[yolo_name] = ExperimentConfig(
            name=yolo_name,
            phase="slicing",
            detector="yolo11m",
            weights="yolo11m.pt",
            resolution="640",
            imgsz=640,
            method=method,
            batch=16,
            accumulation=1,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_bias_lr=0.1,
            amp=True,
            rect=True,
            epochs=150,
            patience=10,
            seed=42,
            confidence=0.001,
        )

        # RT-DETR-L slicing
        rtdetr_name = f"rtdetr-l-slicing-{display}"
        experiments[rtdetr_name] = ExperimentConfig(
            name=rtdetr_name,
            phase="slicing",
            detector="rtdetr-l",
            weights="rtdetr-l.pt",
            resolution="640",
            imgsz=640,
            method=method,
            batch=16,
            accumulation=1,
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.01,
            weight_decay=0.0001,
            warmup_bias_lr=0.0,
            amp=True,
            rect=True,
            epochs=150,
            patience=10,
            seed=42,
            confidence=0.001,
        )

    return experiments


def _build_proposed_experiments() -> dict[str, ExperimentConfig]:
    experiments = {}

    # RT-DETR-L Proposed P2 (2K — Primary Feasibility Checkpoint)
    name_2k = "rtdetr-l-proposed-p2-2k"
    experiments[name_2k] = ExperimentConfig(
        name=name_2k,
        phase="proposed",
        detector="rtdetr-l",
        weights="rtdetr-l.pt",
        resolution="2k",
        imgsz=1920,
        batch=4,
        accumulation=4,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0001,
        warmup_bias_lr=0.0,
        amp=True,
        rect=True,
        epochs=150,
        patience=10,
        seed=42,
        confidence=0.001,
    )

    # RT-DETR-L Proposed P2 (640)
    name_640 = "rtdetr-l-proposed-p2-640"
    experiments[name_640] = ExperimentConfig(
        name=name_640,
        phase="proposed",
        detector="rtdetr-l",
        weights="rtdetr-l.pt",
        resolution="640",
        imgsz=640,
        batch=16,
        accumulation=1,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0001,
        warmup_bias_lr=0.0,
        amp=True,
        rect=True,
        epochs=150,
        patience=10,
        seed=42,
        confidence=0.001,
    )

    # RT-DETR-L Proposed P2 (4K)
    name_4k = "rtdetr-l-proposed-p2-4k"
    experiments[name_4k] = ExperimentConfig(
        name=name_4k,
        phase="proposed",
        detector="rtdetr-l",
        weights="rtdetr-l.pt",
        resolution="4k",
        imgsz=3840,
        batch=2,
        accumulation=8,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0001,
        warmup_bias_lr=0.0,
        amp=False,
        rect=True,
        epochs=150,
        patience=10,
        seed=42,
        confidence=0.001,
    )

    return experiments


# ---------------------------------------------------------------------------
# Complete experiment matrix
# ---------------------------------------------------------------------------

EXPERIMENT_MATRIX: dict[str, ExperimentConfig] = {
    **_build_resolution_experiments(),
    **_build_slicing_experiments(),
    **_build_proposed_experiments(),
}


def resolve_experiment(name: str) -> ExperimentConfig:
    """Resolve an experiment name to its frozen configuration."""
    if name not in EXPERIMENT_MATRIX:
        available = sorted(EXPERIMENT_MATRIX.keys())
        raise ValueError(
            f"Unknown experiment {name!r}.\n"
            f"Available experiments:\n" + "\n".join(f"  - {e}" for e in available)
        )
    return copy.deepcopy(EXPERIMENT_MATRIX[name])


def list_experiments() -> list[dict[str, Any]]:
    """Return a summary of all registered experiments."""
    return [
        {
            "name": cfg.name,
            "phase": cfg.phase,
            "detector": cfg.detector,
            "resolution": cfg.resolution,
            "method": cfg.method if cfg.phase == "slicing" else None,
            "experiment_id": cfg.experiment_id,
        }
        for cfg in EXPERIMENT_MATRIX.values()
    ]
