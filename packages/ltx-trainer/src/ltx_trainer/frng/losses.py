"""Composite training objective hook (paper Eq. 20; L2 + perceptual + Lind — perceptual is optional)."""

from __future__ import annotations

from torch import Tensor, nn


def l2_rendering_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).pow(2).mean()


class PerceptualLossPlaceholder(nn.Module):
    """VGG perceptual loss (Chen & Koltun 2017) is heavy; use L1 on colors for tests or wire LPIPS externally."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).abs().mean()


def frng_composite_loss(
    l2: Tensor,
    l_per: Tensor,
    l_ind: Tensor,
    *,
    lambda_l2: float = 1.0,
    lambda_per: float = 0.5,
) -> Tensor:
    """Eq. (20): ``L = λ1 L2 + λper Lper + Lind`` (``Lind`` already weighted internally)."""
    return lambda_l2 * l2 + lambda_per * l_per + l_ind
