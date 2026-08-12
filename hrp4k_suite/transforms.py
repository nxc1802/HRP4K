from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class CoordinateTransform(Protocol):
    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray: ...
    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray: ...


class IdentityTransform:
    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return np.asarray(boxes_xyxy, dtype=float).copy()

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return np.asarray(boxes_xyxy, dtype=float).copy()


@dataclass(frozen=True)
class CropTransform:
    x0: float
    y0: float

    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] -= self.x0; result[:, [1, 3]] -= self.y0
        return result

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] += self.x0; result[:, [1, 3]] += self.y0
        return result


@dataclass
class SeparableWarpTransform:
    """Monotonic source->warped axis maps used by FOVEA/TPP-style methods."""
    source_x: np.ndarray
    warped_x: np.ndarray
    source_y: np.ndarray
    warped_y: np.ndarray

    def __post_init__(self):
        for values, name in ((self.source_x, "source_x"), (self.warped_x, "warped_x"),
                             (self.source_y, "source_y"), (self.warped_y, "warped_y")):
            values = np.asarray(values, dtype=float)
            if values.ndim != 1 or len(values) < 2 or np.any(np.diff(values) <= 0):
                raise ValueError(f"{name} must be a strictly increasing 1-D map")
            setattr(self, name, values)
        if len(self.source_x) != len(self.warped_x) or len(self.source_y) != len(self.warped_y):
            raise ValueError("source and warped axis maps must have matching lengths")

    @staticmethod
    def _map_boxes(boxes: np.ndarray, src_x, dst_x, src_y, dst_y) -> np.ndarray:
        result = np.asarray(boxes, dtype=float).copy()
        result[:, [0, 2]] = np.interp(result[:, [0, 2]], src_x, dst_x)
        result[:, [1, 3]] = np.interp(result[:, [1, 3]], src_y, dst_y)
        return result

    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return self._map_boxes(boxes_xyxy, self.source_x, self.warped_x, self.source_y, self.warped_y)

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return self._map_boxes(boxes_xyxy, self.warped_x, self.source_x, self.warped_y, self.source_y)


@dataclass
class GridWarpTransform:
    """Dense source/warped point maps for ZoomDet-style non-separable warps."""
    forward_grid: np.ndarray
    inverse_grid: np.ndarray

    def __post_init__(self):
        self.forward_grid = np.asarray(self.forward_grid, dtype=float)
        self.inverse_grid = np.asarray(self.inverse_grid, dtype=float)
        for grid, name in ((self.forward_grid, "forward_grid"), (self.inverse_grid, "inverse_grid")):
            if grid.ndim != 3 or grid.shape[2] != 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
                raise ValueError(f"{name} must have shape [height,width,2]")

    @staticmethod
    def _sample(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
        height, width = grid.shape[:2]
        x = np.clip(points[:, 0], 0, width - 1); y = np.clip(points[:, 1], 0, height - 1)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, width - 1); y1 = np.minimum(y0 + 1, height - 1)
        wx = (x - x0)[:, None]; wy = (y - y0)[:, None]
        top = grid[y0, x0] * (1 - wx) + grid[y0, x1] * wx
        bottom = grid[y1, x0] * (1 - wx) + grid[y1, x1] * wx
        return top * (1 - wy) + bottom * wy

    @classmethod
    def _boxes(cls, grid: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=float)
        corners = np.stack((boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]), axis=1)
        mapped = cls._sample(grid, corners.reshape(-1, 2)).reshape(-1, 4, 2)
        return np.column_stack((mapped[:, :, 0].min(1), mapped[:, :, 1].min(1),
                                mapped[:, :, 0].max(1), mapped[:, :, 1].max(1)))

    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return self._boxes(self.forward_grid, boxes_xyxy)

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        return self._boxes(self.inverse_grid, boxes_xyxy)


@dataclass
class ProcessedView:
    image: np.ndarray
    transform: CoordinateTransform
    source_width: int
    source_height: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def map_box(self, xyxy) -> list[float]:
        source = self.transform.inverse_boxes(np.asarray([xyxy], dtype=float))[0]
        x1, y1, x2, y2 = source
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
