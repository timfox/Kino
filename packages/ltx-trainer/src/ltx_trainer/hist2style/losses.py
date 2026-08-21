"""Perceptual MSE + marginal 1D Wasserstein-2 loss (Algorithm 1, §3.3)."""

from __future__ import annotations

import torch
from torch import Tensor


def marginal_wasserstein_loss(x: Tensor, y: Tensor) -> Tensor:
    """Algorithm 1 — sorted 1D Wasserstein-2 per channel."""
    # x, y: B×C×HW flattened as B×HW×C
    if x.dim() == 4:
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        y = y.reshape(b, c, -1).permute(0, 2, 1)
    losses = []
    for c in range(x.shape[-1]):
        xc = torch.sort(x[..., c], dim=1).values
        yc = torch.sort(y[..., c], dim=1).values
        losses.append(((xc - yc) ** 2).mean())
    return torch.stack(losses).mean()
