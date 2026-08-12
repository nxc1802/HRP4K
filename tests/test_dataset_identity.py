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
        self.assertFalse(result["official_dataset_identity"])
        self.assertFalse(result["annotation_hash_match"]["train"])


if __name__ == "__main__": unittest.main()
