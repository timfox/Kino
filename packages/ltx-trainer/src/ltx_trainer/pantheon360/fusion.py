"""Dual-anchor latent fusion (Sec. 3.6, Time Reversal Fusion)."""

from __future__ import annotations

from torch import Tensor


def dual_anchor_latent_fusion(
    x_fwd: Tensor,
    x_bwd: Tensor,
) -> Tensor:
    """``x_{t-1} = 1/2 (x_{t-1,s} + x_{t-1,e})`` at each denoising step."""
    return 0.5 * (x_fwd + x_bwd)
