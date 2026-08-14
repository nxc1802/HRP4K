from __future__ import annotations

import math
import unittest

from hrp4k_suite.predictions import validate_predictions


GT = {"images": [{"id": 1, "width": 100, "height": 80}], "categories": [{"id": 7}]}
VALID = {"image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4], "score": 0.5}


class PredictionValidationTests(unittest.TestCase):
    def test_valid_prediction_is_normalized(self):
        self.assertEqual(validate_predictions(GT, [VALID])[0]["bbox"], [1.0, 2.0, 3.0, 4.0])

    def test_invalid_records_fail_fast(self):
        invalid = [
            {**VALID, "image_id": 2}, {**VALID, "category_id": 8}, {**VALID, "bbox": [1, 2, -1, 4]},
            {**VALID, "bbox": [1, 2, math.nan, 4]}, {**VALID, "score": math.inf}, {**VALID, "score": 1.1},
            {key: value for key, value in VALID.items() if key != "bbox"}, "not-a-dict",
        ]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError): validate_predictions(GT, [record])

    def test_strict_bounds(self):
        with self.assertRaises(ValueError):
            validate_predictions(GT, [{**VALID, "bbox": [99, 1, 2, 2]}], strict_bounds=True)


if __name__ == "__main__": unittest.main()
