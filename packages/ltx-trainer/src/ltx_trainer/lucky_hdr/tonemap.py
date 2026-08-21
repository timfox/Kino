"""Exposure normalization and μ-law tone mapping (LuckyHDR SI-HDR)."""

from __future__ import annotations

import torch
from torch import Tensor


def tone_map_mu(x: Tensor, *, mu: float = 5000.0) -> Tensor:
    """μ-law tone map on linear ``[0,1]`` radiance."""
    x = x.clamp(min=0.0)
    return torch.log1p(mu * x) / torch.log1p(torch.tensor(mu, device=x.device, dtype=x.dtype))


def normalize_exposure(img: Tensor, ev: float, ref_ev: float = 0.0) -> Tensor:
    """Scale linear RGB by ``2^(ev - ref_ev)`` and clip to ``[0,1]``."""
    scale = 2.0 ** (float(ev) - float(ref_ev))
    return (img * scale).clamp(0.0, 1.0)
