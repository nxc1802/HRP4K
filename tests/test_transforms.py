from __future__ import annotations

import unittest

import numpy as np

from hrp4k_suite.transforms import CropTransform, GridWarpTransform, IdentityTransform, SeparableWarpTransform


class TransformTests(unittest.TestCase):
    def setUp(self): self.boxes = np.asarray([[10, 20, 30, 40]], dtype=float)

    def test_identity_and_crop_round_trip(self):
        for transform in (IdentityTransform(), CropTransform(4, 7)):
            np.testing.assert_allclose(transform.inverse_boxes(transform.forward_boxes(self.boxes)), self.boxes)

    def test_separable_round_trip(self):
        transform = SeparableWarpTransform(np.array([0, 50, 100]), np.array([0, 30, 100]),
                                           np.array([0, 50, 100]), np.array([0, 70, 100]))
        np.testing.assert_allclose(transform.inverse_boxes(transform.forward_boxes(self.boxes)), self.boxes, atol=1e-6)

    def test_identity_grid_round_trip(self):
        y, x = np.mgrid[0:101, 0:101]; grid = np.stack((x, y), axis=-1)
        transform = GridWarpTransform(grid, grid)
        np.testing.assert_allclose(transform.inverse_boxes(transform.forward_boxes(self.boxes)), self.boxes, atol=1e-6)


if __name__ == "__main__": unittest.main()
