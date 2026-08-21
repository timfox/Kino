"""CLEAR training objectives (Sec. 3.4, Eq. 5–6)."""

from __future__ import annotations

import torch
from torch import Tensor


def lsae_reconstruction(
    h: Tensor,
    h_hat: Tensor,
    f: Tensor,
    *,
    sparsity_lambda: float = 1e-4,
) -> Tensor:
    """Eq. (5) single-layer reconstruction + L1 sparsity."""
    rec = torch.nn.functional.mse_loss(h_hat, h)
    sparse = sparsity_lambda * f.abs().mean()
    return rec + sparse


def lcon_separability(s_spe: Tensor, s_uni: Tensor, eps: float = 1e-8) -> Tensor:
    """Eq. (6): log(1 + S_uni / (S_spe + eps))."""
    return torch.log1p(s_uni / (s_spe + eps))
