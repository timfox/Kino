"""Trajectory prediction metrics (Sec. III-D)."""

from __future__ import annotations

import torch
from torch import Tensor


def ade(pred: Tensor, gt: Tensor) -> Tensor:
    """Average displacement error along horizon."""
    return (pred - gt).norm(dim=-1).mean(dim=-1)


def fde(pred: Tensor, gt: Tensor) -> Tensor:
    """Final displacement error."""
    return (pred[..., -1, :] - gt[..., -1, :]).norm(dim=-1)


def min_ade(pred_modes: Tensor, gt: Tensor) -> float:
    """minADE over K modes: (B, K, F, 2) vs (B, F, 2)."""
    errs = ade(pred_modes, gt.unsqueeze(1))
    return float(errs.min(dim=1).values.mean().item())


def min_fde(pred_modes: Tensor, gt: Tensor) -> float:
    errs = fde(pred_modes, gt.unsqueeze(1))
    return float(errs.min(dim=1).values.mean().item())


def mlade(pred_modes: Tensor, gt: Tensor, mode_logits: Tensor) -> float:
    best = mode_logits.argmax(dim=-1)
    idx = best.view(-1, 1, 1, 1).expand(-1, 1, pred_modes.shape[2], pred_modes.shape[3])
    likely = pred_modes.gather(1, idx).squeeze(1)
    return float(ade(likely, gt).mean().item())


def nll_pos(mode_logits: Tensor, pred_modes: Tensor, gt: Tensor) -> float:
    """Stub Gaussian NLL using best-mode residual."""
    probs = torch.softmax(mode_logits, dim=-1)
    best = probs.max(dim=-1).values.clamp(min=1e-6)
    err = ade(pred_modes, gt.unsqueeze(1)).min(dim=1).values
    return float((-torch.log(best) + err).mean().item())
