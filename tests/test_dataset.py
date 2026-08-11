from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hrp4k_suite.dataset import dataset_completeness, prepare_dataset_view


class DatasetTests(unittest.TestCase):
    def test_completeness_requires_train_and_valid_then_test(self):
        manifest = {
            "official_reference": {"train": 4203, "valid": 900, "test": 900},
            "splits": {"train": {"selected_images": 4203}, "valid": {"selected_images": 12}, "test": {"selected_images": 900}},
        }
        self.assertEqual(dataset_completeness(manifest), {"official_training_complete": False, "official_benchmark_complete": False})
        manifest["splits"]["valid"]["selected_images"] = 900
        self.assertEqual(dataset_completeness(manifest), {"official_training_complete": True, "official_benchmark_complete": True})
        manifest["splits"]["test"]["selected_images"] = 12
        self.assertEqual(dataset_completeness(manifest), {"official_training_complete": True, "official_benchmark_complete": False})

    def test_unbounded_view_selects_all_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source"; output = root / "output"
            for split, count in (("train", 3), ("valid", 2), ("test", 1)):
                image_dir = source / split / "images"; image_dir.mkdir(parents=True)
                images = []
                for image_id in range(count):
                    name = f"{image_id}.jpg"; (image_dir / name).write_bytes(b"image")
                    images.append({"id": image_id, "file_name": name, "width": 100, "height": 100})
                payload = {"categories": [{"id": 0, "name": "pothole"}], "images": images, "annotations": []}
                (source / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
            manifest = prepare_dataset_view(source, output)
            self.assertEqual([manifest["splits"][split]["selected_images"] for split in ("train", "valid", "test")], [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
