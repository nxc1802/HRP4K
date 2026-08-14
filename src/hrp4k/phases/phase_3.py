from __future__ import annotations

from pathlib import Path
from typing import Any

from ..diagnostics.diagnostics import diagnose


def run_phase_3(
    ground_truth: Path, prediction_paths: list[Path], output_dir: Path,
) -> dict[str, Any]:
    """Execute Phase 3 deep diagnostics exclusively from saved predictions."""
    return diagnose(ground_truth, prediction_paths, output_dir)
