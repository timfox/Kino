"""Bilateral grid slice + per-pixel affine color transform (§3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def guidance_map(rgb: Tensor) -> Tensor:
    """Learned luminance-like guidance g: R³→R (stub: Rec.601 luma)."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def slice_bilateral_grid(
    grid_affine: Tensor,
    grid_alpha: Tensor,
    content: Tensor,
    g_bins: int,
    h_bins: int,
    w_bins: int,
) -> Tensor:
    """Bilateral grid slice stub: per-pixel affine coeffs + alpha-weighted residual."""
    b, c, h, w = content.shape
    # grid_affine: B×12×Hf×Wf — use first 12 channels as 3×4 affine per pixel
    scale = F.interpolate(grid_affine, size=(h, w), mode="bilinear", align_corners=False)
    alpha = F.interpolate(grid_alpha, size=(h, w), mode="bilinear", align_corners=False).clamp(min=1e-4)
    delta = scale[:, :3] * 0.1
    return (content + delta).clamp(0, 1)


class ChannelLUT(nn.Module):
    """Smooth per-channel monotonic LUT (§3.4)."""

    def __init__(self, bins: int = 256) -> None:
        super().__init__()
        self.curve = nn.Parameter(torch.linspace(0, 1, bins))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        flat = (x.reshape(b * c, -1) * (self.curve.numel() - 1)).long().clamp(0, self.curve.numel() - 1)
        return self.curve[flat].reshape(b, c, h, w)
