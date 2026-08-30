from __future__ import annotations

import unittest

import numpy as np

from hrp4k.methods.base import CropTransform, GridWarpTransform, IdentityTransform, SeparableWarpTransform


class TransformTests(unittest.TestCase):
    def setUp(self): self.boxes = np.asarray([[10, 20, 30, 40]], dtype=float)

    def test_identity_and_crop_round_trip(self):
        for transform in (IdentityTransform(), CropTransform(4, 7)):
            np.testing.assert_allclose(transform.inverse_boxes(transform.forward_boxes(self.boxes)), self.boxes)

    def test_separable_round_trip(self):
        transform = SeparableWarpTransform(np.array([0, 50, 100]), np.array([0, 30, 100]),
                                           np.array([0, 50, 100]), np.array([0, 70, 100]))
        np.testing.assert_allclose(transform.inverse_boxes(transform.forward_boxes(self.boxes)), self.boxes, atol=1e-6)

    def test_randomized_and_boundary_boxes(self):
        rng = np.random.RandomState(42)
        x1 = rng.uniform(0, 40, size=20)
        y1 = rng.uniform(0, 40, size=20)
        x2 = x1 + rng.uniform(1, 40, size=20)
        y2 = y1 + rng.uniform(1, 40, size=20)
        boxes = np.column_stack([x1, y1, x2, y2])
        # Add boundary edge boxes
        boundary_boxes = np.vstack([boxes, [[0.0, 0.0, 10.0, 10.0]], [[50.0, 50.0, 100.0, 100.0]]])

        crop = CropTransform(15.5, 22.3)
        np.testing.assert_allclose(crop.inverse_boxes(crop.forward_boxes(boundary_boxes)), boundary_boxes, atol=1e-6)

        separable = SeparableWarpTransform(np.array([0, 30, 70, 100]), np.array([0, 20, 80, 100]),
                                           np.array([0, 40, 60, 100]), np.array([0, 25, 75, 100]))
        np.testing.assert_allclose(separable.inverse_boxes(separable.forward_boxes(boundary_boxes)), boundary_boxes, atol=1e-5)

    def test_invalid_transform_parameters_raise_errors(self):
        with self.assertRaises(ValueError):
            # non-strictly increasing
            SeparableWarpTransform(np.array([0, 50, 50, 100]), np.array([0, 30, 70, 100]),
                                   np.array([0, 50, 100]), np.array([0, 70, 100]))
        with self.assertRaises(ValueError):
            # mismatched length
            SeparableWarpTransform(np.array([0, 50, 100]), np.array([0, 30, 70, 100, 120]),
                                   np.array([0, 50, 100]), np.array([0, 70, 100]))
        with self.assertRaises(ValueError):
            # bad grid shape
            GridWarpTransform(np.zeros((10, 10)), np.zeros((10, 10)))


if __name__ == "__main__":
    unittest.main()
