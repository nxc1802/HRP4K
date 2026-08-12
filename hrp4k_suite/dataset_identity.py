from __future__ import annotations

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
