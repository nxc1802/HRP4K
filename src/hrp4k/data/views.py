from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .coco import available_image_ids, filtered_coco, load_split
from . import identity as _identity_module
from .paths import SPLITS, image_path


def _balanced_sample(data: dict[str, Any], limit: int, seed: int) -> set[int]:
    rng = random.Random(seed)
    positive = {int(a["image_id"]) for a in data.get("annotations", [])}
    all_ids = {int(im["id"]) for im in data.get("images", [])}
    negative = all_ids - positive
    pos, neg = sorted(positive), sorted(negative)
    rng.shuffle(pos); rng.shuffle(neg)
    pos_target = min(len(pos), max(1, round(limit * 2 / 3)))
    neg_target = min(len(neg), limit - pos_target)
    chosen = pos[:pos_target] + neg[:neg_target]
    if len(chosen) < limit:
        remaining = sorted(all_ids - set(chosen)); rng.shuffle(remaining)
        chosen.extend(remaining[:limit - len(chosen)])
    return set(chosen)


def dataset_completeness(manifest: dict[str, Any]) -> dict[str, bool]:
    reference = manifest["official_reference"]; splits = manifest["splits"]
    training = all(splits[split]["selected_images"] == reference[split] for split in ("train", "valid"))
    benchmark = training and splits["test"]["selected_images"] == reference["test"]
    return {"official_training_complete": training, "official_benchmark_complete": benchmark}


def prepare_dataset_view(
    data_dir: Path, output_dir: Path, train_limit: int | None = None, valid_limit: int | None = None,
    test_limit: int | None = None, seed: int = 42,
) -> dict[str, Any]:
    """Create a deterministic YOLO/COCO view using symlinks; None selects all available images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = {"train": train_limit, "valid": valid_limit, "test": test_limit}
    manifest: dict[str, Any] = {
        "source": str(data_dir.resolve()), "seed": seed, "splits": {},
        "official_reference": {"train": 4203, "valid": 900, "test": 900},
        "annotation_sha256": {},
    }
    for offset, split in enumerate(SPLITS):
        declared = load_split(data_dir, split)
        source = filtered_coco(data_dir, split)
        requested_limit = limits[split]
        selected_limit = len(source["images"]) if requested_limit is None else min(requested_limit, len(source["images"]))
        chosen = _balanced_sample(source, selected_limit, seed + offset)
        sampled = {
            **{key: value for key, value in source.items() if key not in {"images", "annotations"}},
            "images": [im for im in source["images"] if int(im["id"]) in chosen],
            "annotations": [ann for ann in source["annotations"] if int(ann["image_id"]) in chosen],
        }
        split_dir = output_dir / split
        image_dir, label_dir = split_dir / "images", split_dir / "labels"
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        for directory in (image_dir, label_dir):
            for stale in directory.iterdir():
                if stale.is_file() or stale.is_symlink(): stale.unlink()
        annotations = defaultdict(list)
        for ann in sampled["annotations"]:
            annotations[int(ann["image_id"])].append(ann)
        for im in sampled["images"]:
            source_image = image_path(data_dir, split, im["file_name"]).resolve()
            target_image = image_dir / Path(im["file_name"]).name
            if target_image.is_symlink() or target_image.exists():
                target_image.unlink()
            target_image.symlink_to(source_image)
            lines = []
            width, height = float(im["width"]), float(im["height"])
            for ann in annotations[int(im["id"])]:
                x, y, box_w, box_h = map(float, ann["bbox"])
                lines.append(f"0 {(x + box_w / 2) / width:.8f} {(y + box_h / 2) / height:.8f} {box_w / width:.8f} {box_h / height:.8f}")
            (label_dir / f"{target_image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        (output_dir / f"{split}.json").write_text(json.dumps(sampled, indent=2), encoding="utf-8")
        annotation_path = data_dir / f"{split}.json"
        manifest["annotation_sha256"][split] = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
        manifest["splits"][split] = {
            "declared_images": len(declared.get("images", [])), "available_images": len(source["images"]),
            "selected_images": len(sampled["images"]), "annotations": len(sampled["annotations"]),
        }
    yaml_text = (
        f"path: {output_dir.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "names:\n  0: pothole\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    identity = _identity_module.verify_dataset_identity(manifest)
    manifest.update(identity)
    full_release_view = all(
        limits[split] is None
        and manifest["splits"][split]["selected_images"] == manifest["splits"][split]["available_images"]
        for split in SPLITS
    )
    official_view = bool(identity["official_dataset_identity"] and full_release_view)
    manifest.update({
        "official_training_complete": official_view,
        "official_benchmark_complete": official_view,
        "official_dataset_view": official_view,
        "benchmark_label": "official" if official_view else "smoke",
    })
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_smoke_dataset(
    data_dir: Path, output_dir: Path, train_limit: int = 24, valid_limit: int = 12,
    test_limit: int = 12, seed: int = 42,
) -> dict[str, Any]:
    """Compatibility wrapper for a bounded smoke dataset view."""
    return prepare_dataset_view(data_dir, output_dir, train_limit, valid_limit, test_limit, seed)
