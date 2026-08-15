"""Canonical COCO export helper for external runners.

Compatible with Python 3.8+ in isolated external conda/venv environments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] to [x, y, width, height]."""
    x1, y1, x2, y2 = map(float, box)
    return [
        float(x1),
        float(y1),
        float(x2 - x1),
        float(y2 - y1),
    ]


def make_detection(
    image_id: int,
    category_id: int,
    xyxy: Sequence[float],
    score: float,
) -> dict[str, Any]:
    """Create a canonical detection record."""
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": xyxy_to_xywh(xyxy),
        "score": float(score),
    }


def save_prediction_document(
    output_path: str | Path,
    method: str,
    detector: dict[str, Any] | str,
    predictions: list[dict[str, Any]],
    image_metadata: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Save a canonical HRP4K prediction JSON document."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "method": method,
        "detector": detector if isinstance(detector, dict) else {"name": str(detector)},
        "predictions": predictions,
        "image_metadata": image_metadata or [],
        "external_provenance": provenance or {},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
