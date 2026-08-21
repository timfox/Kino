"""Multi-level training objective (Sec. III-E, Eq. 8–11)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def loss_bce(pred: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(pred, target)


def loss_kl(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """LKL — predicted heatmap vs soft GT distribution (Eq. 8)."""
    p = F.softmax(pred.flatten(1), dim=-1).clamp_min(eps)
    t = F.softmax(target.flatten(1), dim=-1).clamp_min(eps)
    return (t * (t.log() - p.log())).sum(dim=-1).mean()


def loss_rtc(
    region_feats: Tensor,
    text_feats: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE region–text contrastive (Eq. 10)."""
    v = F.normalize(region_feats, dim=-1)
    t = F.normalize(text_feats, dim=-1)
    logits = torch.matmul(v, t.transpose(-1, -2)) / temperature
    c = v.shape[-2]
    labels = torch.arange(c, device=v.device)
    if v.dim() == 3:
        b = v.shape[0]
        return F.cross_entropy(logits.reshape(b * c, c), labels.repeat(b))
    return F.cross_entropy(logits, labels)


def total_loss(
    l_bce: Tensor,
    l_kl: Tensor,
    l_rtc: Tensor,
    *,
    lambda_bce: float,
    lambda_kl: float,
    lambda_rtc: float,
) -> Tensor:
    return lambda_bce * l_bce + lambda_kl * l_kl + lambda_rtc * l_rtc
