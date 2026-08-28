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
    forward_grid: np.ndarray  # [canvas_h, canvas_w, 2] -> maps canvas (u, v) to source (x, y)
    inverse_grid: np.ndarray  # [sample_h, sample_w, 2] -> maps source (x, y) to canvas (u, v)
    source_size: tuple[int, int] = (3840, 2160)  # (w, h)
    canvas_size: tuple[int, int] = (640, 640)    # (w, h)

    def __post_init__(self):
        self.forward_grid = np.asarray(self.forward_grid, dtype=float)
        self.inverse_grid = np.asarray(self.inverse_grid, dtype=float)
        for grid, name in ((self.forward_grid, "forward_grid"), (self.inverse_grid, "inverse_grid")):
            if grid.ndim != 3 or grid.shape[2] != 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
                raise ValueError(f"{name} must have shape [height,width,2]")

    @staticmethod
    def _sample(grid: np.ndarray, points: np.ndarray, domain_w: float, domain_h: float) -> np.ndarray:
        gh, gw = grid.shape[:2]
        gx = np.clip(points[:, 0] / max(1.0, domain_w - 1.0) * (gw - 1), 0, gw - 1)
        gy = np.clip(points[:, 1] / max(1.0, domain_h - 1.0) * (gh - 1), 0, gh - 1)
        x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
        x1 = np.minimum(x0 + 1, gw - 1); y1 = np.minimum(y0 + 1, gh - 1)
        wx = (gx - x0)[:, None]; wy = (gy - y0)[:, None]
        top = grid[y0, x0] * (1 - wx) + grid[y0, x1] * wx
        bottom = grid[y1, x0] * (1 - wx) + grid[y1, x1] * wx
        return top * (1 - wy) + bottom * wy

    def _map_boxes(self, grid: np.ndarray, boxes: np.ndarray, domain_w: float, domain_h: float) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=float)
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=float)
        corners = np.stack((boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]), axis=1)
        mapped = self._sample(grid, corners.reshape(-1, 2), domain_w, domain_h).reshape(-1, 4, 2)
        return np.column_stack((mapped[:, :, 0].min(1), mapped[:, :, 1].min(1),
                                mapped[:, :, 0].max(1), mapped[:, :, 1].max(1)))

    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map source 4K boxes -> canvas boxes."""
        return self._map_boxes(self.inverse_grid, boxes_xyxy, self.source_size[0], self.source_size[1])

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map canvas boxes -> source 4K boxes."""
        return self._map_boxes(self.forward_grid, boxes_xyxy, self.canvas_size[0], self.canvas_size[1])


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
        w = max(0.01, float(x2 - x1))
        h = max(0.01, float(y2 - y1))
        return [float(x1), float(y1), w, h]


def nms(predictions: list[dict[str, Any]], threshold: float = 0.5) -> tuple[list[dict[str, Any]], int]:
    if not predictions:
        return [], 0
    boxes = np.asarray([p["bbox"] for p in predictions], dtype=float)
    xyxy = np.column_stack((boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]))
    scores = np.asarray([p["score"] for p in predictions], dtype=float)
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        current = int(order[0]); keep.append(current)
        if order.size == 1: break
        rest = order[1:]
        x1 = np.maximum(xyxy[current, 0], xyxy[rest, 0]); y1 = np.maximum(xyxy[current, 1], xyxy[rest, 1])
        x2 = np.minimum(xyxy[current, 2], xyxy[rest, 2]); y2 = np.minimum(xyxy[current, 3], xyxy[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_current = boxes[current, 2] * boxes[current, 3]
        area_rest = boxes[rest, 2] * boxes[rest, 3]
        overlaps = inter / np.maximum(area_current + area_rest - inter, 1e-12)
        order = rest[overlaps <= threshold]
    return [predictions[index] for index in keep], len(predictions) - len(keep)


def _starts(length: int, window: int, overlap: float) -> list[int]:
    if window >= length:
        return [0]
    stride = max(1, round(window * (1 - overlap)))
    starts = list(range(0, max(1, length - window + 1), stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


METHOD_REGISTRY = {
    "resize": {"type": "inference", "requires_training": False, "implementation": "native", "status": "ready"},
    "uniform-2": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "uniform-3": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "sliced-nms": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "sahi": {"type": "crop", "requires_training": False, "implementation": "official-library", "status": "optional-ready"},
    "perspective-grid": {"type": "crop", "requires_training": False, "implementation": "native", "status": "ready"},
    "autofocus": {"type": "coarse-to-fine", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "adazoom": {"type": "adaptive-crop", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "fovea": {"type": "nonlinear-warp", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "two-plane-prior": {"type": "nonlinear-warp", "requires_training": True, "implementation": "paper-reproduction", "status": "external-required"},
    "zoomdet": {"type": "nonlinear-warp", "requires_training": False, "implementation": "native", "status": "ready"},
    "zoomdet-geometry": {"type": "nonlinear-warp", "requires_training": False, "implementation": "road-geometry-prior", "status": "ready"},
    "zoomdet-neural": {"type": "nonlinear-warp", "requires_training": True, "implementation": "official-neural-network", "status": "ready"},
    "adapoth": {"type": "adaptive-region-scout", "requires_training": True, "implementation": "adapoth-lite-core", "status": "ready"},
    "adapoth-lite": {"type": "adaptive-region-scout", "requires_training": True, "implementation": "adapoth-lite-core", "status": "ready"},
    "adapoth-oracle": {"type": "adaptive-region-scout", "requires_training": False, "implementation": "oracle-upper-bound", "status": "ready"},
    "adapoth-fixed": {"type": "adaptive-region-scout", "requires_training": True, "implementation": "fixed-k-ablation", "status": "ready"},
    "adapoth-random": {"type": "adaptive-region-scout", "requires_training": False, "implementation": "random-crop-ablation", "status": "ready"},
}

METHOD_STATUS = {
    name: f"{entry['status']} ({entry['implementation']})"
    for name, entry in METHOD_REGISTRY.items()
}


def make_views(
    image,
    method: str,
    tile_size: int = 960,
    overlap: float = 0.2,
    scout_weights: Any | None = None,
    context_margin: float = 0.20,
    k_max: int = 4,
    gt_boxes_4k: list[list[float]] | None = None,
    device: str | None = None,
) -> list[ProcessedView]:
    import warnings
    height, width = image.shape[:2]
    if method == "sahi":
        raise ValueError("Official SAHI is executed by the generic runner, not make_views()")
    if method == "resize":
        return [ProcessedView(image, IdentityTransform(), width, height)]
    if method.startswith("adapoth"):
        from .adapoth import make_adapoth_views
        views, _ = make_adapoth_views(
            image=image,
            method=method,
            scout_weights=scout_weights,
            threshold=0.30,
            context_margin=context_margin,
            k_max=k_max,
            crop_size=(640, 640),
            global_size=(960, 544),
            gt_boxes_4k=gt_boxes_4k,
            device=device,
        )
        return views
    if method in {"zoomdet", "zoomdet-geometry"}:
        from .zoomdet import make_zoomdet_view
        return [make_zoomdet_view(image, canvas_size=tile_size if tile_size <= 1280 else 640, mode="geometry")]
    if method == "zoomdet-neural":
        from .zoomdet import make_zoomdet_view
        return [make_zoomdet_view(image, canvas_size=tile_size if tile_size <= 1280 else 640, mode="neural")]
    if method.startswith("uniform"):
        grid = int(method.split("-", 1)[1]) if "-" in method else 2
        views = []
        for row in range(grid):
            y0, y1 = round(row * height / grid), round((row + 1) * height / grid)
            for col in range(grid):
                x0, x1 = round(col * width / grid), round((col + 1) * width / grid)
                views.append(ProcessedView(image[y0:y1, x0:x1], CropTransform(x0, y0), x1 - x0, y1 - y0,
                                           {"crop": [x0, y0, x1, y1]}))
        return views
    if method == "sliced-nms":
        views = []
        window_w = min(tile_size, width)
        window_h = min(max(1, round(tile_size * height / width)), height)
        for y0 in _starts(height, window_h, overlap):
            for x0 in _starts(width, window_w, overlap):
                views.append(ProcessedView(image[y0:y0 + window_h, x0:x0 + window_w], CropTransform(x0, y0), window_w, window_h,
                                           {"crop": [x0, y0, x0 + window_w, y0 + window_h]}))
        return views
    if method == "perspective-bands":
        warnings.warn("'perspective-bands' removes vertical context but does not magnify horizontally; use 'perspective-grid'", DeprecationWarning)
        boundaries = [0, round(height * 0.45), round(height * 0.72), height]
        return [ProcessedView(image[y0:y1], CropTransform(0, y0), width, y1 - y0) for y0, y1 in zip(boundaries, boundaries[1:])]
    if method == "perspective-grid":
        # Hand-designed ground-plane baseline with 2D (horizontal + vertical) overlap.
        # Far bands receive more horizontal crops and therefore more detector pixels.
        boundaries = [0, round(height * 0.45), round(height * 0.72), height]
        columns_by_band = [4, 3, 2]
        views = []
        for idx, ((y0, y1), columns) in enumerate(zip(zip(boundaries, boundaries[1:]), columns_by_band)):
            band_h = y1 - y0
            pad_y = round(band_h * overlap * 0.5)
            y0_crop = max(0, y0 - pad_y) if idx > 0 else 0
            y1_crop = min(height, y1 + pad_y) if idx < len(columns_by_band) - 1 else height
            crop_h = y1_crop - y0_crop

            window_w = min(width, int(np.ceil(width / (columns - (columns - 1) * overlap))))
            starts = _starts(width, window_w, overlap)
            if len(starts) > columns:
                starts = np.linspace(0, width - window_w, columns, dtype=int).tolist()
            for x0 in starts:
                views.append(ProcessedView(image[y0_crop:y1_crop, x0:x0 + window_w],
                                           CropTransform(x0, y0_crop), window_w, crop_h,
                                           {"crop": [x0, y0_crop, x0 + window_w, y1_crop]}))
        return views
    raise ValueError(f"Unknown processing method: {method}")
