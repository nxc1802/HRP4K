"""HRP4K Warped Dataset Generation Pipeline for ZoomDet Training.

Warps native 4K images and their ground truth bounding boxes onto continuous deformation
canvases (e.g. 640x640) for end-to-end ZoomDet detector training.
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
from ..methods.zoomdet import generate_deformation_grid, warp_image


def create_warped_dataset(
    data_dir: Path | str,
    output_dir: Path | str,
    canvas_size: int = 640,
    horizon_ratio: float = 0.40,
    splits: tuple[str, ...] = ("train", "valid"),
) -> dict[str, Any]:
    """Generate warped training dataset from 4K source images and annotations for ZoomDet.
    
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

    summary: dict[str, Any] = {"canvas_size": canvas_size, "horizon_ratio": horizon_ratio, "splits": {}}

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
        ann_id_counter = 0
        processed_images = 0

        for img_entry in images:
            img_path = image_path(data_dir, split, img_entry["file_name"])
            if not img_path.is_file():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            remap_x, remap_y, transform = generate_deformation_grid(
                source_h=h, source_w=w, canvas_h=canvas_size, canvas_w=canvas_size, horizon_ratio=horizon_ratio
            )

            # 1. Warp image to canvas
            warped_img = warp_image(img, remap_x, remap_y, canvas_size=(canvas_size, canvas_size))
            warped_filename = img_entry["file_name"]
            warped_filepath = img_out_dir / warped_filename
            lbl_filepath = lbl_out_dir / f"{Path(warped_filename).stem}.txt"

            cv2.imwrite(str(warped_filepath), warped_img)

            # 2. Map bounding boxes onto warped canvas
            img_anns = img_to_anns.get(int(img_entry["id"]), [])
            yolo_lines = []

            if img_anns:
                boxes_xyxy = []
                for ann in img_anns:
                    gx, gy, gw, gh = map(float, ann["bbox"])
                    boxes_xyxy.append([gx, gy, gx + gw, gy + gh])

                mapped_boxes = transform.forward_boxes(np.array(boxes_xyxy, dtype=float))

                for ann, mapped in zip(img_anns, mapped_boxes):
                    u1, v1, u2, v2 = mapped
                    # Clip to canvas boundaries
                    u1 = max(0.0, min(float(canvas_size), u1))
                    v1 = max(0.0, min(float(canvas_size), v1))
                    u2 = max(0.0, min(float(canvas_size), u2))
                    v2 = max(0.0, min(float(canvas_size), v2))

                    bw = max(0.01, u2 - u1)
                    bh = max(0.01, v2 - v1)

                    if bw >= 2 and bh >= 2:
                        cx = (u1 + bw / 2.0) / canvas_size
                        cy = (v1 + bh / 2.0) / canvas_size
                        norm_w = bw / canvas_size
                        norm_h = bh / canvas_size

                        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

                        ann_id_counter += 1
                        coco_annotations.append({
                            "id": ann_id_counter,
                            "image_id": img_entry["id"],
                            "category_id": categories[0]["id"],
                            "bbox": [u1, v1, bw, bh],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                        })

            lbl_filepath.write_text("\n".join(yolo_lines), encoding="utf-8")
            processed_images += 1

            coco_images.append({
                "id": img_entry["id"],
                "file_name": warped_filename,
                "width": canvas_size,
                "height": canvas_size,
            })

        # Save COCO JSON
        split_coco = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }
        (output_dir / f"{split}.json").write_text(json.dumps(split_coco, indent=2), encoding="utf-8")

        summary["splits"][split] = {
            "processed_images": processed_images,
            "annotations": ann_id_counter,
        }

    summary["dataset_type"] = "warped"
    summary["official_dataset_identity"] = True
    summary["official_dataset_view"] = True

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
