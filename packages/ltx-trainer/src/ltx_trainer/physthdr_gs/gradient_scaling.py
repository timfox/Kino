"""Illumination-guided gradient scaling (Sec. 4.3, Eq. 17–18)."""

from __future__ import annotations

import torch
from torch import Tensor


def scaling_factor(la: Tensor, la_hat: Tensor, *, s: float = 1.0) -> Tensor:
    """
    Per-Gaussian scaling sa = s · σ(|La − ˆLa|) + 1 (Eq. 17).

    Args:
        la, la_hat: [N, 3] illumination tensors
    Returns:
        sa [N, 1]
    """
    delta = (la - la_hat).abs().mean(dim=-1, keepdim=True)
    return s * torch.sigmoid(delta) + 1.0


def scaled_gradient_norm(grad: Tensor, sa: Tensor) -> Tensor:
    """Apply scaling factor to gradient magnitude for densification gate."""
    if grad.dim() == 1:
        grad = grad.unsqueeze(-1)
    return (sa * grad.norm(dim=-1, keepdim=True)).mean()
