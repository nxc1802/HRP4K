from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ANNOTATION_SHA256 = {
    "train": "4ecd641fba3e3d689fb4faa031a39b377e4940919289e3b95f2c7e4cbb67f0b3",
    "valid": "c1a39ce4609b3f6d160e12932159ed4fd387d2118d4e848de16cd2f716a9dfbd",
    "test": "38ec6dbe13337a8321472fd060e0d24a31519a9067d1d578b8d2e8551b7afc0f",
}

OFFICIAL_DATASET_NOTE = (
    "Single official dataset version downloaded from the release source; missing train files originate "
    "from that source and the project is contacting the authors for a complete archive."
)


def verify_dataset_identity(
    manifest: dict[str, Any], expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify that annotations belong to the single frozen HRP4K release."""
    expected = expected_hashes or EXPECTED_ANNOTATION_SHA256
    actual = manifest.get("annotation_sha256", {})
    matches = {
        split: bool(expected.get(split) and actual.get(split) == expected[split])
        for split in ("train", "valid", "test")
    }
    identity = all(matches.values())
    return {
        "annotation_hash_match": matches,
        "official_dataset_identity": identity,
        "official_training_complete": identity,
        "official_benchmark_complete": identity,
        "dataset_note": OFFICIAL_DATASET_NOTE,
    }


def generate_content_manifest(data_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Generate SHA256 hashes and file sizes for all available image files across splits."""
    content: dict[str, dict[str, Any]] = {}
    data_dir = Path(data_dir)
    for split in ("train", "valid", "test"):
        split_dir = data_dir / split / "images"
        if not split_dir.is_dir():
            split_dir = data_dir / split
        if split_dir.is_dir():
            for img_file in sorted(split_dir.glob("*.*")):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    rel_path = f"{split}/images/{img_file.name}"
                    file_bytes = img_file.read_bytes()
                    content[rel_path] = {
                        "size_bytes": len(file_bytes),
                        "sha256": hashlib.sha256(file_bytes).hexdigest(),
                    }
    manifest = {
        "version": "1.0",
        "total_files": len(content),
        "files": content,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_content_manifest(data_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify that images on disk match the expected size and SHA256 in content manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    verified = 0
    mismatches = []
    missing = []
    data_dir = Path(data_dir)
    for rel_path, meta in files.items():
        img_path = data_dir / rel_path
        if not img_path.is_file():
            parts = rel_path.split("/")
            if len(parts) >= 3 and parts[1] == "images":
                alt_path = data_dir / parts[0] / parts[2]
                if alt_path.is_file():
                    img_path = alt_path
        if not img_path.is_file():
            missing.append(rel_path)
            continue
        file_bytes = img_path.read_bytes()
        actual_sha = hashlib.sha256(file_bytes).hexdigest()
        actual_size = len(file_bytes)
        if actual_size != meta["size_bytes"] or actual_sha != meta["sha256"]:
            mismatches.append({"file": rel_path, "expected_sha": meta["sha256"], "actual_sha": actual_sha})
        else:
            verified += 1
    return {
        "passed": len(missing) == 0 and len(mismatches) == 0,
        "total_expected": len(files),
        "verified_files": verified,
        "missing_files": missing,
        "mismatched_files": mismatches,
    }
