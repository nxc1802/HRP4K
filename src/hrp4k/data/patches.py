"""HRP4K Patch Generation and Sliced-Dataset Preparation Pipeline.

Generates high-resolution tile datasets (e.g. 640x640) from native 4K images for
Patch-based training (Crop Before Training).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ..data.coco import load_split
from ..data.paths import image_path
from ..methods.base import _starts


def box_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def crop_annotations_for_patch(
    annotations: list[dict[str, Any]],
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    min_visibility: float = 0.25,
) -> list[dict[str, Any]]:
    """Crop and filter bounding boxes relative to the patch coordinates."""
    patch_boxes = []
    crop_x1 = crop_x0 + crop_w
    crop_y1 = crop_y0 + crop_h

    for ann in annotations:
        gx, gy, gw, gh = map(float, ann["bbox"])
        gx2, gy2 = gx + gw, gy + gh
        orig_area = max(1e-6, gw * gh)

        # Intersection with crop
        ix1 = max(crop_x0, gx)
        iy1 = max(crop_y0, gy)
        ix2 = min(crop_x1, gx2)
        iy2 = min(crop_y1, gy2)

        if ix2 > ix1 and iy2 > iy1:
            inter_w = ix2 - ix1
            inter_h = iy2 - iy1
            inter_area = inter_w * inter_h
            visibility = inter_area / orig_area

            if visibility >= min_visibility and inter_w >= 3 and inter_h >= 3:
                # Relative coordinates inside patch
                rel_x = ix1 - crop_x0
                rel_y = iy1 - crop_y0
                patch_boxes.append({
                    "category_id": ann.get("category_id", 0),
                    "bbox": [rel_x, rel_y, inter_w, inter_h],
                    "visibility": float(visibility),
                    "original_id": ann.get("id"),
                })

    return patch_boxes


def create_patch_dataset(
    data_dir: Path | str,
    output_dir: Path | str,
    tile_size: int = 640,
    overlap: float = 0.2,
    bg_ratio: float = 0.20,
    min_visibility: float = 0.25,
    splits: tuple[str, ...] = ("train", "valid"),
) -> dict[str, Any]:
    """Generate patch-based training dataset from 4K source images and annotations.
    
    Creates:
      - output_dir/train/images, output_dir/train/labels (.txt YOLO format)
      - output_dir/valid/images, output_dir/valid/labels (.txt YOLO format)
      - output_dir/train.json, output_dir/valid.json (COCO JSON format)
      - output_dir/dataset.yaml (Ultralytics Training YAML)
      - output_dir/manifest.json (Dataset metadata)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"tile_size": tile_size, "overlap": overlap, "splits": {}}

    for split in splits:
        split_data = load_split(data_dir, split)
        images = split_data.get("images", [])
        annotations = split_data.get("annotations", [])
        categories = split_data.get("categories", [{"id": 0, "name": "pothole"}])

        img_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in annotations:
            img_id = int(ann["image_id"])
            img_to_anns.setdefault(img_id, []).append(ann)

        img_out_dir = output_dir / split / "images"
        lbl_out_dir = output_dir / split / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []
        patch_id_counter = 0
        ann_id_counter = 0
        total_positive_patches = 0
        total_negative_patches = 0

        for img_entry in images:
            img_path = image_path(data_dir, split, img_entry["file_name"])
            if not img_path.is_file():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            img_anns = img_to_anns.get(int(img_entry["id"]), [])

            # Generate grid starts
            starts_y = _starts(h, tile_size, overlap)
            starts_x = _starts(w, tile_size, overlap)

            image_positive_patches = []
            image_negative_patches = []

            for y0 in starts_y:
                for x0 in starts_x:
                    patch_w = min(tile_size, w - x0)
                    patch_h = min(tile_size, h - y0)
                    patch_img = img[y0:y0 + patch_h, x0:x0 + patch_w]

                    # Pad to tile_size if boundary slice is smaller
                    if patch_w < tile_size or patch_h < tile_size:
                        canvas = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
                        canvas[:patch_h, :patch_w] = patch_img
                        patch_img = canvas

                    patch_boxes = crop_annotations_for_patch(
                        img_anns, x0, y0, patch_w, patch_h, min_visibility=min_visibility
                    )

                    patch_info = {
                        "patch_img": patch_img,
                        "x0": x0, "y0": y0, "w": patch_w, "h": patch_h,
                        "boxes": patch_boxes,
                    }

                    if patch_boxes:
                        image_positive_patches.append(patch_info)
                    else:
                        image_negative_patches.append(patch_info)

            # Subsample background patches according to bg_ratio
            num_pos = len(image_positive_patches)
            num_neg_keep = max(1, int(num_pos * bg_ratio)) if num_pos > 0 else 1
            if len(image_negative_patches) > num_neg_keep:
                indices = np.linspace(0, len(image_negative_patches) - 1, num_neg_keep, dtype=int)
                selected_neg = [image_negative_patches[i] for i in indices]
            else:
                selected_neg = image_negative_patches

            selected_patches = image_positive_patches + selected_neg

            stem = Path(img_entry["file_name"]).stem
            for p_idx, p_data in enumerate(selected_patches):
                patch_id_counter += 1
                patch_filename = f"{stem}_patch_{p_idx:03d}.jpg"
                patch_filepath = img_out_dir / patch_filename
                lbl_filepath = lbl_out_dir / f"{stem}_patch_{p_idx:03d}.txt"

                # Save patch image
                cv2.imwrite(str(patch_filepath), p_data["patch_img"])

                # Write YOLO format label
                yolo_lines = []
                for box in p_data["boxes"]:
                    bx, by, bw, bh = box["bbox"]
                    # Normalize relative to tile_size
                    cx = (bx + bw / 2.0) / tile_size
                    cy = (by + bh / 2.0) / tile_size
                    norm_w = bw / tile_size
                    norm_h = bh / tile_size
                    cat_id = 0  # single class pothole
                    yolo_lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

                    # COCO entry
                    ann_id_counter += 1
                    coco_annotations.append({
                        "id": ann_id_counter,
                        "image_id": patch_id_counter,
                        "category_id": categories[0]["id"],
                        "bbox": [bx, by, bw, bh],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                    })

                lbl_filepath.write_text("\n".join(yolo_lines), encoding="utf-8")

                if p_data["boxes"]:
                    total_positive_patches += 1
                else:
                    total_negative_patches += 1

                coco_images.append({
                    "id": patch_id_counter,
                    "file_name": patch_filename,
                    "width": tile_size,
                    "height": tile_size,
                    "original_image_id": img_entry["id"],
                    "crop": [p_data["x0"], p_data["y0"], p_data["w"], p_data["h"]],
                })

        # Write split COCO JSON
        split_coco = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }
        (output_dir / f"{split}.json").write_text(json.dumps(split_coco, indent=2), encoding="utf-8")

        summary["splits"][split] = {
            "source_images": len(images),
            "total_patches": patch_id_counter,
            "positive_patches": total_positive_patches,
            "negative_patches": total_negative_patches,
            "annotations": ann_id_counter,
        }

    # Generate dataset.yaml for Ultralytics training
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": {0: "pothole"},
    }
    with open(output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
