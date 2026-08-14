from __future__ import annotations

import unittest

from hrp4k_suite.dataset_identity import verify_dataset_identity


class DatasetIdentityTests(unittest.TestCase):
    def test_matching_hashes_are_official(self):
        expected = {"train": "a", "valid": "b", "test": "c"}
        result = verify_dataset_identity({"annotation_sha256": expected}, expected)
        self.assertTrue(result["official_dataset_identity"])
        self.assertTrue(all(result["annotation_hash_match"].values()))

    def test_hash_mismatch_is_not_official(self):
        result = verify_dataset_identity(
            {"annotation_sha256": {"train": "bad", "valid": "b", "test": "c"}},
            {"train": "a", "valid": "b", "test": "c"},
        )
    def test_generate_and_verify_content_manifest(self):
        import tempfile
        from pathlib import Path
        from hrp4k_suite.dataset_identity import generate_content_manifest, verify_content_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            img_dir = root / "train" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            (img_dir / "sample.jpg").write_bytes(b"dummy image content")

            manifest_path = root / "dataset_content_manifest.json"
            manifest = generate_content_manifest(root, manifest_path)
            self.assertEqual(manifest["total_files"], 1)
            self.assertIn("train/images/sample.jpg", manifest["files"])

            verification = verify_content_manifest(root, manifest_path)
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["verified_files"], 1)

            # Test mismatch on altered content
            (img_dir / "sample.jpg").write_bytes(b"corrupted content")
            bad_verification = verify_content_manifest(root, manifest_path)
            self.assertFalse(bad_verification["passed"])
            self.assertEqual(len(bad_verification["mismatched_files"]), 1)


if __name__ == "__main__":
    unittest.main()
