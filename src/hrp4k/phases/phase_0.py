from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data.audit import analyze_dataset
from ..data.views import prepare_dataset_view


def run_phase_0(data_dir: Path, output_dir: Path, quality_samples: int = 12) -> dict[str, Any]:
    """Execute Phase 0 dataset integrity and statistical analysis."""
    return analyze_dataset(data_dir, output_dir, quality_samples=quality_samples)
