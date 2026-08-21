"""MemFT objectives (Eq. 8–10)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.parametric_memory.phase_transition import L_CRIT, memft_soft_weight, memft_threshold_weight


def sequence_loss(token_losses: Tensor) -> Tensor:
    return token_losses.mean()


def memft_loss(
    token_losses: Tensor,
    weights: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """L_MemFT = Σ w_t L_t / (Σ w_t + ε) (Eq. 8)."""
    w = weights.clamp(min=0.0)
    return (w * token_losses).sum() / (w.sum() + eps)


def memft_ot_loss(token_losses: Tensor, l_crit: float = L_CRIT) -> Tensor:
    w = memft_threshold_weight(token_losses, l_crit)
    return memft_loss(token_losses, w)


def memft_sw_seq_weights(
    token_losses: Tensor,
    anchor: int,
    *,
    l_crit: float = L_CRIT,
    tau: float = 8.0,
    l_win: int = 32,
    eps_floor: float = 0.01,
) -> Tensor:
    """Eq. 10 spatial sliding weights."""
    base = memft_soft_weight(token_losses, l_crit)
    t_idx = torch.arange(token_losses.numel(), device=token_losses.device, dtype=token_losses.dtype)
    phi = torch.exp(-torch.clamp(t_idx - anchor, min=0.0) / tau)
    w = base.clone()
    mask = t_idx < anchor + l_win
    w = torch.where(mask, base * phi, torch.full_like(w, eps_floor))
    return w


def first_failure_index(pred: Tensor, target: Tensor) -> int:
    diff = pred != target
    if not diff.any():
        return pred.numel()
    return int(diff.nonzero(as_tuple=True)[0][0].item())
