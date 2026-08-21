"""ERP distortion map (Sec. 3.1, Eq. 6)."""

from __future__ import annotations

import torch
from torch import Tensor


def erp_distortion_map(height: int, width: int, *, device: torch.device | None = None) -> Tensor:
    """
    D = cos((h + 0.5 - H/2) * π / H), shape (1, H, W).

    Darker in distortion visualization = larger stretch at poles.
    """
    h_idx = torch.arange(height, dtype=torch.float32, device=device)
    d_row = torch.cos((h_idx + 0.5 - height / 2) * torch.pi / height)
    return d_row.view(1, 1, height, 1).expand(1, 1, height, width).contiguous()


def stretching_ratio_erp(height: int, width: int, *, device: torch.device | None = None) -> Tensor:
    """ρ_ERP = cos(φ) with φ from row index (Eq. 5 proxy)."""
    return erp_distortion_map(height, width, device=device)
