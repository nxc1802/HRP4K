from __future__ import annotations

import math
from typing import Any


REQUIRED_PREDICTION_FIELDS = {"image_id", "category_id", "bbox", "score"}


def validate_predictions(
    gt: dict[str, Any], predictions: list[dict[str, Any]], *, strict_bounds: bool = False,
) -> list[dict[str, Any]]:
    """Validate and normalize canonical COCO detections; never silently repair bad records."""
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    image_by_id = {int(image["id"]): image for image in gt.get("images", [])}
    category_ids = {int(category["id"]) for category in gt.get("categories", [])}
    clean: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f"prediction[{index}] must be an object")
        missing = REQUIRED_PREDICTION_FIELDS - prediction.keys()
        if missing:
            raise ValueError(f"prediction[{index}] missing fields: {sorted(missing)}")
        try:
            image_id = int(prediction["image_id"])
            category_id = int(prediction["category_id"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"prediction[{index}] image_id/category_id must be integers") from exc
        if image_id not in image_by_id:
            raise ValueError(f"prediction[{index}] has unknown image_id={image_id}")
        if category_id not in category_ids:
            raise ValueError(f"prediction[{index}] has unknown category_id={category_id}")
        bbox = prediction["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"prediction[{index}] bbox must be [x,y,w,h]")
        try:
            x, y, width, height = map(float, bbox)
            score = float(prediction["score"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"prediction[{index}] bbox and score must be numeric") from exc
        if not all(math.isfinite(value) for value in (x, y, width, height, score)):
            raise ValueError(f"prediction[{index}] contains NaN/Inf")
        if width <= 0 or height <= 0:
            raise ValueError(f"prediction[{index}] bbox must have positive width/height")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"prediction[{index}] score must be in [0,1]")
        if strict_bounds:
            image = image_by_id[image_id]
            if x < 0 or y < 0 or x + width > float(image["width"]) or y + height > float(image["height"]):
                raise ValueError(f"prediction[{index}] bbox outside image bounds")
        clean.append({**prediction, "image_id": image_id, "category_id": category_id,
                      "bbox": [x, y, width, height], "score": score})
    return clean
