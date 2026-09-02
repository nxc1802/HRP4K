from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


def _unwrap_sequential(model: nn.Module) -> tuple[nn.Module, nn.Sequential, set[int]]:
    """Unwraps model container to retrieve the DetectionModel and its sequential layer list."""
    curr = model
    det_model = None
    seq = None

    if hasattr(curr, "model"):
        if isinstance(curr.model, nn.Sequential):
            det_model = curr
            seq = curr.model
        elif hasattr(curr.model, "model") and isinstance(curr.model.model, nn.Sequential):
            det_model = curr.model
            seq = curr.model.model

    if seq is None:
        if isinstance(model, nn.Sequential):
            seq = model
            det_model = model
        else:
            raise ValueError(f"Could not locate nn.Sequential layer graph in {type(model)}")

    save_indices = getattr(det_model, "save", set())
    if isinstance(save_indices, (list, tuple)):
        save_indices = set(save_indices)

    return det_model, seq, save_indices


def find_c2_backbone_stage(
    model: nn.Module,
    input_size: tuple[int, int] = (640, 640),
    device: torch.device | str = "cpu",
) -> tuple[int, int]:
    """Dynamically discover the backbone layer producing C2 (stride 4) feature map.

    Inspects intermediate layer outputs at runtime without hard-coding layer indices
    or channel dimensions.

    Returns:
        tuple[int, int]: (layer_index, in_channels) for the C2 stage.
    """
    det_model, sub_modules, save_indices = _unwrap_sequential(model)

    dummy = torch.zeros(1, 3, *input_size, device=device)
    x = dummy
    y: list[Any] = []
    c2_idx = None
    c2_channels = None

    was_training = det_model.training
    det_model.eval()

    with torch.no_grad():
        for i, m in enumerate(sub_modules):
            f = getattr(m, "f", -1)
            if f != -1:
                if isinstance(f, int):
                    x = y[f]
                else:
                    x = [x if j == -1 else y[j] for j in f]
            x = m(x)
            y.append(x if i in save_indices else None)

            if isinstance(x, torch.Tensor) and x.ndim == 4:
                stride_h = input_size[0] // x.shape[-2]
                stride_w = input_size[1] // x.shape[-1]
                # Stride 4 stage in early backbone (before downsampling to stride 8 / Stage 3)
                if stride_h == 4 and stride_w == 4 and i < 6:
                    c2_idx = i
                    c2_channels = x.shape[1]

    if was_training:
        det_model.train()

    if c2_idx is None or c2_channels is None:
        raise RuntimeError(
            f"Could not dynamically discover stride-4 C2 backbone stage in model {type(model)}."
        )

    return c2_idx, c2_channels


def extract_c2_backbone(model: nn.Module, x: torch.Tensor, c2_layer_idx: int = 1) -> torch.Tensor:
    """Direct fast extraction of C2 feature map from the backbone without running the full decoder."""
    _, sub_modules, _ = _unwrap_sequential(model)
    curr = x
    for i in range(c2_layer_idx + 1):
        m = sub_modules[i]
        curr = m(curr)
    return curr


class P2Adapter(nn.Module):
    """P2 Adapter module: C2 -> 1x1 Conv -> 3x3 Conv -> P2.

    Projects runtime C2 feature map into standardized P2 feature dimension.
    No hard-coded channel dimensions; derived from model runtime.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act()

        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, c2: torch.Tensor) -> torch.Tensor:
        """Forward pass: C2 -> 1x1 Conv -> 3x3 Conv -> P2."""
        out = self.act1(self.bn1(self.conv1x1(c2)))
        p2 = self.act2(self.bn2(self.conv3x3(out)))
        return p2


class P2Branch(nn.Module):
    """Encapsulates C2 extraction and P2 feature adaptation."""

    def __init__(
        self,
        c2_layer_idx: int,
        in_channels: int,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.c2_layer_idx = c2_layer_idx
        self.adapter = P2Adapter(in_channels=in_channels, out_channels=out_channels)

    def forward(self, c2_feat: torch.Tensor) -> torch.Tensor:
        return self.adapter(c2_feat)
