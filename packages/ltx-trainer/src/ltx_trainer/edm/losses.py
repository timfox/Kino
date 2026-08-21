"""Angular regression + certainty BCE (Eq. 13–16)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def depth_consistency_certainty(
    depth_a_to_b: Tensor,
    depth_b: Tensor,
    *,
    alpha: float = 0.05,
) -> Tensor:
    """Eq. 13 ground-truth certainty mask."""
    rel = (depth_a_to_b - depth_b).abs() / depth_b.clamp_min(1e-6)
    return (rel < alpha).float()


def angular_regression_loss(
    pred_s: Tensor,
    tgt_s: Tensor,
    certainty: Tensor,
) -> Tensor:
    """Eq. 14: cosine similarity on unit sphere."""
    pred = pred_s / pred_s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    tgt = tgt_s / tgt_s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (pred * tgt).sum(dim=-1)
    return (certainty * (1.0 - cos)).sum() / certainty.sum().clamp_min(1.0)


def certainty_bce(pred_c: Tensor, tgt_c: Tensor) -> Tensor:
    """Eq. 15."""
    return F.binary_cross_entropy(pred_c.clamp(1e-6, 1 - 1e-6), tgt_c)


def total_loss(l_r: Tensor, l_c: Tensor, *, lam: float = 0.01) -> Tensor:
    return l_r + lam * l_c
