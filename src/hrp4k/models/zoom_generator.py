"""Official Neural Zoom Generator Module for ZoomDet (CVPR 2023).

Implements a lightweight convolutional neural network that takes a downsampled thumbnail
and predicts an adaptive 2D continuous deformation sampling grid (offset field).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..methods.base import GridWarpTransform, ProcessedView


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NeuralZoomGenerator(nn.Module):
    """Lightweight 2D Deformation Grid Generator Network.
    
    Architecture:
      Input (Thumbnail, e.g. 128x128) -> 4 ConvDown blocks -> 2 ConvUp blocks -> 1x1 Conv (dx, dy)
    """
    def __init__(self, max_displacement: float = 0.35, canvas_size: int = 640):
        super().__init__()
        self.max_displacement = max_displacement
        self.canvas_size = canvas_size

        # Encoder (Downsampling)
        self.enc1 = ConvBlock(3, 16, stride=2)    # 128 -> 64
        self.enc2 = ConvBlock(16, 32, stride=2)   # 64 -> 32
        self.enc3 = ConvBlock(32, 64, stride=2)   # 32 -> 16
        self.enc4 = ConvBlock(64, 128, stride=2)  # 16 -> 8

        # Decoder (Upsampling towards canvas size)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
        )
        # Final head to predict normalized offset field (dx, dy)
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Tanh(),  # Normalized offset in [-1, 1]
        )

    def forward(self, thumbnail: torch.Tensor) -> torch.Tensor:
        """Predict regularized sampling grid.
        
        Args:
            thumbnail: Tensor of shape (B, 3, H, W) normalized to [0, 1].
        Returns:
            grid: Tensor of shape (B, canvas_H, canvas_W, 2) in normalized [-1, 1] coords.
        """
        b = thumbnail.shape[0]
        # Base regular grid in [-1, 1]
        device = thumbnail.device
        base_y, base_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.canvas_size, device=device),
            torch.linspace(-1.0, 1.0, self.canvas_size, device=device),
            indexing="ij",
        )
        base_grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

        # Predict displacement offset field
        feat = self.enc1(thumbnail)
        feat = self.enc2(feat)
        feat = self.enc3(feat)
        feat = self.enc4(feat)
        feat = self.up1(feat)
        feat = self.up2(feat)
        
        # Upsample to canvas resolution
        feat_canvas = F.interpolate(feat, size=(self.canvas_size, self.canvas_size), mode="bilinear", align_corners=False)
        offset = self.head(feat_canvas) * self.max_displacement  # (B, 2, canvas_H, canvas_W)
        offset = offset.permute(0, 2, 3, 1)  # (B, canvas_H, canvas_W, 2)

        # Warp grid = base_grid + offset, clamped to valid [-1, 1] boundary
        grid = torch.clamp(base_grid + offset, -1.0, 1.0)
        return grid


def load_neural_zoom_generator(
    weights_path: Path | str | None = None,
    canvas_size: int = 640,
    device: str = "cpu",
) -> NeuralZoomGenerator:
    """Load and initialize Neural Zoom Generator."""
    model = NeuralZoomGenerator(canvas_size=canvas_size)
    if weights_path and Path(weights_path).is_file():
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        except Exception:
            pass
    model.to(device)
    model.eval()
    return model


def make_neural_zoomdet_view(
    image: np.ndarray,
    generator: NeuralZoomGenerator | None = None,
    canvas_size: int = 640,
    device: str = "cpu",
) -> ProcessedView:
    """Generate a Neural ZoomDet ProcessedView using a lightweight sub-network."""
    if generator is None:
        generator = load_neural_zoom_generator(canvas_size=canvas_size, device=device)

    h, w = image.shape[:2]
    import cv2
    thumbnail = cv2.resize(image, (128, 128))
    thumb_tensor = torch.from_numpy(thumbnail).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    thumb_tensor = thumb_tensor.to(device)

    with torch.no_grad():
        grid = generator(thumb_tensor)  # (1, canvas_size, canvas_size, 2) in [-1, 1]
        
        # Warp image tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)
        warped_tensor = F.grid_sample(img_tensor, grid, mode="bilinear", padding_mode="border", align_corners=False)
        
        warped_img = (warped_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        grid_np = grid[0].cpu().numpy()  # [canvas_h, canvas_w, 2] in [-1, 1]

    # Convert normalized [-1, 1] grid to pixel source coordinates [0, w-1] x [0, h-1]
    fwd_x = ((grid_np[:, :, 0] + 1.0) / 2.0) * (w - 1)
    fwd_y = ((grid_np[:, :, 1] + 1.0) / 2.0) * (h - 1)
    forward_grid = np.stack((fwd_x, fwd_y), axis=-1).astype(np.float32)

    # Build inverse grid [sample_h, sample_w, 2]
    sample_h, sample_w = 256, 256
    u_norm = np.linspace(0, 1, canvas_size)
    v_norm = np.linspace(0, 1, canvas_size)
    inv_y_norm = np.linspace(0, 1, sample_h)
    inv_x_norm = np.linspace(0, 1, sample_w)
    
    # Fast 1D/2D interpolation on dense samples
    src_y_col = fwd_y[:, 0]
    canvas_v = np.interp(inv_y_norm * (h - 1), np.sort(src_y_col), v_norm) * (canvas_size - 1)
    
    inv_x_grid = np.zeros((sample_h, sample_w), dtype=np.float32)
    inv_y_grid = np.zeros((sample_h, sample_w), dtype=np.float32)
    for r, v_val in enumerate(canvas_v):
        inv_y_grid[r, :] = v_val
        row_idx = int(np.clip(round(v_val), 0, canvas_size - 1))
        fwd_x_row = fwd_x[row_idx, :]
        inv_u = np.interp(inv_x_norm * (w - 1), np.sort(fwd_x_row), u_norm * (canvas_size - 1))
        inv_x_grid[r, :] = inv_u

    inverse_grid = np.stack((inv_x_grid, inv_y_grid), axis=-1).astype(np.float32)
    transform = GridWarpTransform(
        forward_grid=forward_grid,
        inverse_grid=inverse_grid,
        source_size=(w, h),
        canvas_size=(canvas_size, canvas_size),
    )

    return ProcessedView(
        image=warped_img,
        transform=transform,
        source_width=w,
        source_height=h,
        metadata={"method": "zoomdet-neural", "canvas_size": canvas_size},
    )
