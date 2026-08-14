from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .paths import image_path, SPLITS


def scale_class(area_ratio: float) -> str:
    if area_ratio < 0.0005:
        return "ultra_fine"
    if area_ratio < 0.001:
        return "fine"
    if area_ratio < 0.0025:
        return "medium"
    return "large"


SCALE_ORDER = ("ultra_fine", "fine", "medium", "large")


def load_split(data_dir: Path, split: str) -> dict[str, Any]:
    path = data_dir / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def available_image_ids(data_dir: Path, split: str) -> set[int]:
    data = load_split(data_dir, split)
    return {
        int(im["id"]) for im in data.get("images", [])
        if image_path(data_dir, split, im["file_name"]).is_file()
    }


def filtered_coco(data_dir: Path, split: str, image_ids: set[int] | None = None) -> dict[str, Any]:
    """Return a COCO document containing only images that physically exist."""
    data = load_split(data_dir, split)
    usable = available_image_ids(data_dir, split)
    if image_ids is not None:
        usable &= {int(value) for value in image_ids}
    return {
        **{key: value for key, value in data.items() if key not in {"images", "annotations"}},
        "images": [im for im in data.get("images", []) if int(im["id"]) in usable],
        "annotations": [ann for ann in data.get("annotations", []) if int(ann["image_id"]) in usable],
    }


def iter_master_rows(data_dir: Path, splits: Iterable[str] = SPLITS):
    for split in splits:
        data = load_split(data_dir, split)
        images = {int(im["id"]): im for im in data.get("images", [])}
        counts = Counter(int(a["image_id"]) for a in data.get("annotations", []))
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            im = images.get(image_id)
            if not im:
                continue
            width, height = float(im["width"]), float(im["height"])
            x, y, box_w, box_h = map(float, ann["bbox"])
            ratio = box_w * box_h / (width * height)
            yield {
                "split": split,
                "image_id": image_id,
                "annotation_id": int(ann["id"]),
                "file_name": Path(im["file_name"]).name,
                "image_available": image_path(data_dir, split, im["file_name"]).is_file(),
                "image_width": int(width),
                "image_height": int(height),
                "x": x,
                "y": y,
                "width_px": box_w,
                "height_px": box_h,
                "x_center": (x + box_w / 2) / width,
                "y_center": (y + box_h / 2) / height,
                "y_bottom": (y + box_h) / height,
                "width_rel": box_w / width,
                "height_rel": box_h / height,
                "area_ratio": ratio,
                "log_area": math.log10(max(ratio, 1e-12)),
                "aspect_ratio": box_w / box_h if box_h else 0.0,
                "scale_class": scale_class(ratio),
                "objects_in_image": counts[image_id],
            }
