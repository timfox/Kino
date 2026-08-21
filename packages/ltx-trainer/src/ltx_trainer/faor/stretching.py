"""Pixel-wise stretching ratio map Md (Eq. 5)."""

from __future__ import annotations

import torch
from torch import Tensor


def stretching_ratio_map(h: int, w: int, *, device: torch.device | None = None) -> Tensor:
    """Md(h,w) = 255 * cos((h+0.5 - H/2)/H * pi)."""
    rows = torch.arange(h, device=device, dtype=torch.float32)
    phi = (rows + 0.5 - h / 2) / h * torch.pi
    md_row = 255.0 * torch.cos(phi)
    return md_row.view(h, 1).expand(h, w)
