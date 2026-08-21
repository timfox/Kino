"""Disparity → depth conversion (Sec. 2.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def disparity_to_depth(
    disparity_hat: Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-3,
    depth_min: float = 0.0,
    depth_max: float = 10.0,
) -> Tensor:
    """
    d = 1 / (α σ(d̂) + ε), clipped to [depth_min, depth_max] meters.
    """
    d = 1.0 / (alpha * torch.sigmoid(disparity_hat) + eps)
    return d.clamp(depth_min, depth_max)


def relative_depth_stub(erp: Tensor, *, scale: float = 1.0) -> Tensor:
    """Cheap monocular depth prior for smoke (center-biased)."""
    b, _, h, w = erp.shape
    v = torch.linspace(0, 1, h, device=erp.device, dtype=erp.dtype).view(1, 1, h, 1)
    u = torch.linspace(0, 1, w, device=erp.device, dtype=erp.dtype).view(1, 1, 1, w)
    base = 1.5 + 2.0 * (0.5 - (v - 0.5).abs()) + 0.5 * u
    return (base * scale).expand(b, 1, h, w).squeeze(1)
