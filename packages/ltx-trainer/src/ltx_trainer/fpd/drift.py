"""Lifted multi-bandwidth drift loss (Eq. 6–7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def laplace_affinity(x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
    """Doubly-normalized softmax affinity a_ij ∝ exp(−‖x−y‖/h)."""
    dist = (x - y).norm(dim=-1, keepdim=True)
    raw = torch.exp(-dist / max(bandwidth, 1e-6))
    raw = raw / (raw.sum(dim=0, keepdim=True) + 1e-8)
    return raw / (raw.sum(dim=1, keepdim=True) + 1e-8)


def drift_vector(
    x: Tensor,
    y_pos: Tensor,
    x_neg: Tensor,
    bandwidth: float,
) -> Tensor:
    """Eq. (6): attractive to teacher targets + repulsive from in-batch negatives."""
    # x, y_pos, x_neg: [B, D] flattened spatial features
    b = x.shape[0]
    v = torch.zeros_like(x)
    for i in range(b):
        att = torch.zeros_like(x[i])
        for j in range(b):
            a = laplace_affinity(x[i : i + 1], y_pos[j : j + 1], bandwidth).squeeze()
            att = att + a * (y_pos[j] - x[i])
        rep = torch.zeros_like(x[i])
        for j in range(b):
            a = laplace_affinity(x[i : i + 1], x_neg[j : j + 1], bandwidth).squeeze()
            rep = rep + a * (x_neg[j] - x[i])
        v[i] = att - rep
    return v


def drift_loss(
    student_features: Tensor,
    teacher_features: Tensor,
    bandwidths: tuple[float, ...],
) -> Tensor:
    """Eq. (7): L_drift over bandwidth set H with in-batch cyclic negatives."""
    b = student_features.shape[0]
    x_neg = torch.roll(student_features.detach(), shifts=1, dims=0)
    total = torch.tensor(0.0, device=student_features.device)
    for h in bandwidths:
        v = drift_vector(student_features, teacher_features, x_neg, h)
        target = (student_features + v).detach()
        zh = student_features.norm() + 1e-8
        loss_h = F.mse_loss(student_features, target) / zh
        total = total + loss_h
    return total / max(1, len(bandwidths))
