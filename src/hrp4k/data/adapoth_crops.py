"""AdaPoth-Lite Local Crop Training Dataset Generation Pipeline.

Supports:
1. Stage 2 (Local Crop Pretraining):
   - 50% Positive local crops (center jitter, scale jitter, context expansion, flip, color jitter)
   - 25% Hard negatives (road cracks, tar repairs, shadows, water, concrete joints from empty areas)
   - 25% Full-image downscaled samples
2. Stage 3 (Scout-Generated Crop Fine-Tuning):
   - 60% Scout-generated candidate crops (matching inference distribution)
   - 40% GT local crops and full-image samples
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ..data.coco import load_split
from ..data.paths import image_path
from ..models.scout import CandidateGenerator, MobileNetV3Scout


def _normalize_box(box_xywh: list[float] | tuple[float, ...], img_w: int, img_h: int) -> list[float]:
    x, y, w, h = box_xywh[:4]
    cx = (x + w * 0.5) / float(img_w)
    cy = (y + h * 0.5) / float(img_h)
    nw = w / float(img_w)
    nh = h / float(img_h)
    return [float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)),
            float(np.clip(nw, 0.0, 1.0)), float(np.clip(nh, 0.0, 1.0))]


def _remap_boxes_to_crop(
    orig_boxes: list[list[float]],
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    min_visibility: float = 0.20,
) -> list[list[float]]:
    """Map 4K GT boxes into crop relative coordinates."""
    crop_boxes = []
    crop_x1 = crop_x0 + crop_w
    crop_y1 = crop_y0 + crop_h

    for box in orig_boxes:
        gx, gy, gw, gh = box[:4]
        gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
        orig_area = max(1e-6, gw * gh)

        ix1 = max(crop_x0, gx)
        iy1 = max(crop_y0, gy)
        ix2 = min(crop_x1, gx2)
        iy2 = min(crop_y1, gy2)

        if ix2 > ix1 and iy2 > iy1:
            inter_w = ix2 - ix1
            inter_h = iy2 - iy1
            inter_area = inter_w * inter_h
            if (inter_area / orig_area) >= min_visibility and inter_w >= 4 and inter_h >= 4:
                crop_boxes.append([ix1 - crop_x0, iy1 - crop_y0, inter_w, inter_h])

    return crop_boxes


def create_adapoth_crop_dataset(
    data_dir: Path | str,
    output_dir: Path | str,
    stage: str = "stage2",  # "stage2" or "stage3"
    scout_weights: Path | str | None = None,
    target_crop_size: tuple[int, int] = (640, 640),
    context_margin: float = 0.20,
    seed: int = 42,
    splits: tuple[str, ...] = ("train", "valid"),
) -> dict[str, Any]:
    """Generate publication-grade training dataset for YOLO11n-P2-lite."""
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scout_model = None
    candidate_gen = None
    if stage == "stage3":
        if not scout_weights or not Path(scout_weights).is_file():
            raise ValueError(f"Stage 3 crop generation requires a valid trained Scout checkpoint (--scout-weights {scout_weights})")
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scout_model = MobileNetV3Scout().to(dev)
        ckpt = torch.load(str(scout_weights), map_location=dev, weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        scout_model.load_state_dict(state_dict)
        scout_model.eval()
        candidate_gen = CandidateGenerator(threshold=0.25, context_margin=context_margin, k_max=6)

    manifest_splits: dict[str, Any] = {}

    for split in splits:
        split_img_dir = output_dir / "images" / split
        split_lbl_dir = output_dir / "labels" / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        coco = load_split(data_dir, split)
        images = [im for im in coco.get("images", []) if image_path(data_dir, split, im["file_name"]).is_file()]
        img_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            img_to_anns.setdefault(int(ann["image_id"]), []).append(ann)

        generated_samples = 0
        positive_crops = 0
        hard_negatives = 0
        full_samples = 0
        scout_crops = 0

        for im_info in images:
            img_id = int(im_info["id"])
            file_p = image_path(data_dir, split, im_info["file_name"])
            img = cv2.imread(str(file_p))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            anns = img_to_anns.get(img_id, [])
            boxes = [list(map(float, a["bbox"])) for a in anns]
            base_stem = Path(im_info["file_name"]).stem

            if stage == "stage2":
                # 1. Full-image sample (25% weight)
                full_img_out = split_img_dir / f"{base_stem}_full.jpg"
                full_lbl_out = split_lbl_dir / f"{base_stem}_full.txt"
                cv2.imwrite(str(full_img_out), cv2.resize(img, target_crop_size))
                lines = []
                for b in boxes:
                    norm = _normalize_box(b, orig_w, orig_h)
                    lines.append(f"0 {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")
                full_lbl_out.write_text("\n".join(lines), encoding="utf-8")
                full_samples += 1

                # 2. Positive local crops (50% weight)
                for b_idx, b in enumerate(boxes):
                    bx, by, bw, bh = b
                    # Apply center & scale jitter
                    jitter_cx = (bx + bw * 0.5) + random.uniform(-0.2, 0.2) * bw
                    jitter_cy = (by + bh * 0.5) + random.uniform(-0.2, 0.2) * bh
                    crop_w = int(max(320, bw * (1.0 + context_margin * 2.0) * random.uniform(0.9, 1.3)))
                    crop_h = int(max(240, bh * (1.0 + context_margin * 2.0) * random.uniform(0.9, 1.3)))
                    crop_w = min(orig_w, crop_w)
                    crop_h = min(orig_h, crop_h)

                    cx0 = int(np.clip(jitter_cx - crop_w * 0.5, 0, orig_w - crop_w))
                    cy0 = int(np.clip(jitter_cy - crop_h * 0.5, 0, orig_h - crop_h))

                    crop = img[cy0:cy0 + crop_h, cx0:cx0 + crop_w]
                    crop_boxes = _remap_boxes_to_crop(boxes, cx0, cy0, crop_w, crop_h)

                    c_img_out = split_img_dir / f"{base_stem}_pos_{b_idx}.jpg"
                    c_lbl_out = split_lbl_dir / f"{base_stem}_pos_{b_idx}.txt"
                    cv2.imwrite(str(c_img_out), cv2.resize(crop, target_crop_size))

                    c_lines = []
                    for cb in crop_boxes:
                        cnorm = _normalize_box(cb, crop_w, crop_h)
                        c_lines.append(f"0 {cnorm[0]:.6f} {cnorm[1]:.6f} {cnorm[2]:.6f} {cnorm[3]:.6f}")
                    c_lbl_out.write_text("\n".join(c_lines), encoding="utf-8")
                    positive_crops += 1

                # 3. Hard Negative crops (25% weight - patches from road area without potholes)
                num_hard_neg = max(1, len(boxes) // 2)
                for n_idx in range(num_hard_neg):
                    neg_w = random.randint(480, 800)
                    neg_h = random.randint(360, 600)
                    # Sample in lower 60% of image (road surface)
                    neg_x0 = random.randint(0, max(0, orig_w - neg_w))
                    neg_y0 = random.randint(int(orig_h * 0.40), max(int(orig_h * 0.40), orig_h - neg_h))
                    neg_boxes = _remap_boxes_to_crop(boxes, neg_x0, neg_y0, neg_w, neg_h)
                    if len(neg_boxes) == 0:  # Pure negative
                        neg_crop = img[neg_y0:neg_y0 + neg_h, neg_x0:neg_x0 + neg_w]
                        n_img_out = split_img_dir / f"{base_stem}_neg_{n_idx}.jpg"
                        n_lbl_out = split_lbl_dir / f"{base_stem}_neg_{n_idx}.txt"
                        cv2.imwrite(str(n_img_out), cv2.resize(neg_crop, target_crop_size))
                        n_lbl_out.write_text("", encoding="utf-8")  # Empty label file
                        hard_negatives += 1

            elif stage == "stage3":
                # Stage 3: Scout-generated candidate crops (60%) + Full/GT (40%)
                import torch
                # Generate candidate regions from Scout
                thumb = cv2.resize(img, (960, 540))
                thumb_t = torch.from_numpy(thumb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                thumb_t = (thumb_t - mean) / std

                with torch.no_grad():
                    dev = next(scout_model.parameters()).device
                    hmap = scout_model(thumb_t.to(dev)).cpu().numpy()[0, 0]

                candidates = candidate_gen.generate(hmap, source_width=orig_w, source_height=orig_h)

                # Save Scout candidate crops
                for c_idx, cand in enumerate(candidates):
                    crop_w = cand.width
                    crop_h = cand.height
                    crop = img[cand.y0:cand.y1, cand.x0:cand.x1]
                    if crop.size == 0:
                        continue
                    crop_boxes = _remap_boxes_to_crop(boxes, cand.x0, cand.y0, crop_w, crop_h)

                    sc_img_out = split_img_dir / f"{base_stem}_scout_{c_idx}.jpg"
                    sc_lbl_out = split_lbl_dir / f"{base_stem}_scout_{c_idx}.txt"
                    cv2.imwrite(str(sc_img_out), cv2.resize(crop, target_crop_size))

                    sc_lines = []
                    for cb in crop_boxes:
                        cnorm = _normalize_box(cb, crop_w, crop_h)
                        sc_lines.append(f"0 {cnorm[0]:.6f} {cnorm[1]:.6f} {cnorm[2]:.6f} {cnorm[3]:.6f}")
                    sc_lbl_out.write_text("\n".join(sc_lines), encoding="utf-8")
                    scout_crops += 1

                # Add full image sample
                full_img_out = split_img_dir / f"{base_stem}_full.jpg"
                full_lbl_out = split_lbl_dir / f"{base_stem}_full.txt"
                cv2.imwrite(str(full_img_out), cv2.resize(img, target_crop_size))
                lines = []
                for b in boxes:
                    norm = _normalize_box(b, orig_w, orig_h)
                    lines.append(f"0 {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")
                full_lbl_out.write_text("\n".join(lines), encoding="utf-8")
                full_samples += 1

        manifest_splits[split] = {
            "total_images": len(images),
            "positive_crops": positive_crops,
            "hard_negatives": hard_negatives,
            "full_samples": full_samples,
            "scout_crops": scout_crops,
            "total_generated": positive_crops + hard_negatives + full_samples + scout_crops,
        }

    # Generate dataset.yaml for Ultralytics
    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "names": {0: "pothole"},
    }
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    manifest = {
        "dataset_type": f"adapoth_{stage}",
        "stage": stage,
        "source_dataset": str(data_dir),
        "target_crop_size": list(target_crop_size),
        "splits": manifest_splits,
        "scout_weights": str(scout_weights) if scout_weights else None,
        "dataset_yaml": str((output_dir / "dataset.yaml").resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "dataset_yaml": str(output_dir / "dataset.yaml"),
        "manifest": manifest,
    }
