"""ZoomDet: 2D Continuous Deformation Grid for High-Resolution Object Detection (CVPR 2023).

Generates a continuous 2D non-uniform deformation grid that magnifies high-frequency road pothole
regions while compressing background and sky regions on a single fixed canvas (e.g. 640x640).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import GridWarpTransform, ProcessedView


_GRID_CACHE: dict[tuple[int, int, int, int, float], tuple[np.ndarray, np.ndarray, GridWarpTransform]] = {}


def generate_deformation_grid(
    source_h: int,
    source_w: int,
    canvas_h: int = 640,
    canvas_w: int = 640,
    horizon_ratio: float = 0.40,
    road_expansion: float = 1.75,
    sky_compression: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, GridWarpTransform]:
    """Generate dense forward and inverse deformation grids.
    
    Returns:
        (remap_x, remap_y, GridWarpTransform)
    """
    cache_key = (source_h, source_w, canvas_h, canvas_w, horizon_ratio)
    if cache_key in _GRID_CACHE:
        remap_x, remap_y, transform = _GRID_CACHE[cache_key]
        return remap_x, remap_y, transform
    # 1. Vertical non-linear mapping
    v_norm = np.linspace(0, 1, canvas_h)
    
    # Sigmoidal / piecewise smooth cumulative density mapping
    # Sky occupies [0, 0.20] of canvas, but maps to [0, horizon_ratio] of source
    # Road occupies [0.20, 1.0] of canvas, mapping to [horizon_ratio, 1.0] of source
    canvas_split = 0.20
    src_y_norm = np.zeros_like(v_norm)
    
    sky_mask = v_norm < canvas_split
    road_mask = ~sky_mask
    
    # Sky region: linear / smooth cubic
    src_y_norm[sky_mask] = (v_norm[sky_mask] / canvas_split) * horizon_ratio
    
    # Road region: non-linear quadratic expansion near horizon
    road_t = (v_norm[road_mask] - canvas_split) / (1.0 - canvas_split)
    # Power curve to magnify far potholes while transitioning to near road
    road_curve = np.power(road_t, 1.25)
    src_y_norm[road_mask] = horizon_ratio + road_curve * (1.0 - horizon_ratio)
    
    src_y = src_y_norm * (source_h - 1)

    # 2. Horizontal non-linear perspective mapping (widens center lane)
    u_norm = np.linspace(0, 1, canvas_w)
    src_x_grid = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    src_y_grid = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for r, y_val in enumerate(src_y):
        src_y_grid[r, :] = y_val
        # Depth factor (0 at horizon, 1 at bottom)
        depth = max(0.0, (y_val - horizon_ratio * source_h) / ((1.0 - horizon_ratio) * source_h + 1e-6))
        
        # Center expansion factor increases with distance (near horizon)
        center_magnify = 1.0 + (1.0 - depth) * 0.40
        u_centered = (u_norm - 0.5) * 2.0  # [-1, 1]
        u_warped = np.sign(u_centered) * np.power(np.abs(u_centered), 1.0 / center_magnify)
        u_source = (u_warped / 2.0 + 0.5) * (source_w - 1)
        src_x_grid[r, :] = np.clip(u_source, 0, source_w - 1)

    # Build forward grid [canvas_h, canvas_w, 2] (maps canvas coordinate -> source coordinate)
    forward_grid = np.stack((src_x_grid, src_y_grid), axis=-1)

    # 3. Dense Inverse Grid [source_h_samples, source_w_samples, 2] (maps source coordinate -> canvas coordinate)
    sample_h = 256
    sample_w = 256
    inv_y_norm = np.linspace(0, 1, sample_h)
    inv_x_norm = np.linspace(0, 1, sample_w)
    
    # Invert vertical mapping
    canvas_v = np.interp(inv_y_norm, src_y_norm, v_norm) * (canvas_h - 1)
    
    # Invert horizontal mapping on regular mesh
    inv_x_grid = np.zeros((sample_h, sample_w), dtype=np.float32)
    inv_y_grid = np.zeros((sample_h, sample_w), dtype=np.float32)
    
    for r, v_val in enumerate(canvas_v):
        inv_y_grid[r, :] = v_val
        # Sample forward horizontal curve at row r
        row_idx = int(np.clip(round(v_val), 0, canvas_h - 1))
        fwd_x_row = src_x_grid[row_idx, :]
        inv_u = np.interp(inv_x_norm * (source_w - 1), fwd_x_row, u_norm * (canvas_w - 1))
        inv_x_grid[r, :] = inv_u

    inverse_grid = np.stack((inv_x_grid, inv_y_grid), axis=-1)
    transform = GridWarpTransform(
        forward_grid=forward_grid,
        inverse_grid=inverse_grid,
        source_size=(source_w, source_h),
        canvas_size=(canvas_w, canvas_h),
    )
    _GRID_CACHE[cache_key] = (src_x_grid, src_y_grid, transform)
    return src_x_grid, src_y_grid, transform


def warp_image(
    image: np.ndarray,
    remap_x: np.ndarray,
    remap_y: np.ndarray,
    canvas_size: tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Warp input 4K image to continuous deformation canvas using bilinear interpolation."""
    return cv2.remap(image, remap_x, remap_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def make_zoomdet_view(
    image: np.ndarray,
    canvas_size: int = 640,
    horizon_ratio: float = 0.40,
) -> ProcessedView:
    """Generate a single 2D Continuous Deformation ZoomDet ProcessedView for 1-pass inference."""
    h, w = image.shape[:2]
    remap_x, remap_y, transform = generate_deformation_grid(
        source_h=h, source_w=w, canvas_h=canvas_size, canvas_w=canvas_size, horizon_ratio=horizon_ratio
    )
    warped = warp_image(image, remap_x, remap_y, canvas_size=(canvas_size, canvas_size))
    return ProcessedView(
        image=warped,
        transform=transform,
        source_width=w,
        source_height=h,
        metadata={"method": "zoomdet", "canvas_size": canvas_size, "horizon_ratio": horizon_ratio},
    )
