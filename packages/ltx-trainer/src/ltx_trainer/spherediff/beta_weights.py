"""Foreground beta kernel weights (Eq. 13–14, Sec. B.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def beta_kernel_weights(h: int, w: int, *, b: float = -3.0, device: torch.device | None = None) -> Tensor:
    """
    B(x; b) = (1 - x)^4 * exp(b), x = min(||u||/d_max, 1).
    Flatter than exponential; emphasizes foreground region.
    """
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt(gx * gx + gy * gy)
    d_max = dist.max().clamp_min(1e-6)
    x = (dist / d_max).clamp(0.0, 1.0)
    return (1.0 - x).pow(4) * torch.exp(torch.tensor(b, device=device))
