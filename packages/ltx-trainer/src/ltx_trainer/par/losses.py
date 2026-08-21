"""PAR training objectives (Eq. 6–8)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.par.config import PARConfig


def vanilla_mar_loss(
    noise: Tensor,
    pred: Tensor,
    mask: Tensor,
) -> Tensor:
    """L_va: MSE on masked tokens only (Eq. 6)."""
    m = mask
    if m.dim() == noise.dim() - 1:
        m = m.unsqueeze(-1)
    err = (noise - pred).pow(2)
    return (m * err).sum() / m.sum().clamp_min(1e-6)


def consistency_loss(
    pred: Tensor,
    pred_shifted: Tensor,
    mask_shifted: Tensor,
) -> Tensor:
    """L_consistency on masked regions (Eq. 7)."""
    m = mask_shifted
    if m.dim() == pred.dim() - 1:
        m = m.unsqueeze(-1)
    err = (pred_shifted - pred).pow(2)
    return (m * err).sum() / m.sum().clamp_min(1e-6)


def total_loss(
    l_va: Tensor,
    l_cons: Tensor,
    cfg: PARConfig,
) -> Tensor:
    """L = L_va + λ L_consistency (Eq. 8)."""
    return l_va + cfg.lambda_consistency * l_cons
