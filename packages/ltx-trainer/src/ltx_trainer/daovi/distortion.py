"""ERP distortion map (Sec. 3.3, Eq. 10)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def erp_distortion_weight(height: int, device: torch.device | None = None) -> Tensor:
    """Eq. (10): werp(i,j) = cos((j + 0.5 - N/2) * π / N) — depends on row index only."""
    j = torch.arange(height, device=device, dtype=torch.float32)
    return torch.cos((j + 0.5 - height / 2.0) * math.pi / height)


def distortion_map(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """[1, 1, H, W] map — bright near equator, dark at poles (Fig. 5)."""
    w_row = erp_distortion_weight(height, device=device)
    return w_row.view(1, 1, height, 1).expand(1, 1, height, width)


class DistortionGuidanceGenerator(torch.nn.Module):
    """DGG stub [34]: encode distortion map → guidance for DCN weighting."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, dmap: Tensor) -> Tensor:
        return self.enc(dmap)
