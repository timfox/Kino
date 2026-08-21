"""LatentHDR evaluation helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def luminance(rgb: Tensor) -> Tensor:
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


def dynamic_range_stops(hdr_linear: Tensor, *, p_low: float = 0.01, p_high: float = 0.99) -> float:
    """Estimate dynamic range in stops from luminance percentiles."""
    y = luminance(hdr_linear.clamp(min=0.0)).reshape(-1)
    with torch.no_grad():
        lo = torch.quantile(y.detach(), p_low).clamp(min=1e-8)
        hi = torch.quantile(y.detach(), p_high).clamp(min=1e-8)
        return float(math.log2(hi / lo))
