"""Rectified flow / CFM (Sec. 3.1, Eq. 1–3)."""

from __future__ import annotations

import torch
from torch import Tensor


def rectified_interpolate(z0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    """z_t = (1 - t) z_0 + t ε (Eq. 1)."""
    while t.dim() < z0.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * z0 + t * eps


def cfm_loss(v_pred: Tensor, z0: Tensor, eps: Tensor) -> Tensor:
    """L_CFM = ||v_Θ(z_t, t) - (ε - z_0)||^2 (Eq. 3 target field)."""
    target = eps - z0
    return torch.nn.functional.mse_loss(v_pred, target)
