"""Weighted Spherically Smooth-L1 loss (Eq. 7)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.s3po.config import WSS_L1_BETA
from ltx_trainer.s3po.erp_weights import wss_distortion_weights


def wss_l1_loss(
    pred: Tensor,
    target: Tensor,
    *,
    beta: float = WSS_L1_BETA,
) -> Tensor:
    """WSS-L1 over Y channel [B,1,H,W] or [B,3,H,W] (uses channel mean if RGB)."""
    if pred.shape[1] == 3:
        pred = pred.mean(dim=1, keepdim=True)
        target = target.mean(dim=1, keepdim=True)
    h, w = pred.shape[-2:]
    psi = wss_distortion_weights(h, w, device=pred.device)
    diff = (pred - target).abs()
    smooth = torch.where(
        diff < beta,
        0.5 * diff.pow(2) / beta,
        diff - 0.5 * beta,
    )
    w = psi / psi.sum().clamp_min(1e-6)
    return (smooth * w).sum()
