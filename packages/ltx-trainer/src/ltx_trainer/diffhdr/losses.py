"""Rectified flow matching loss (Sec. 3.3, Eq. 4–5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FlowMatchingConfig:
    pass


def sample_flow_pair(x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Sample x0 ~ N(0,I), t ~ U(0,1), xt = t x1 + (1-t) x0."""
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    while t.dim() < x1.dim():
        t = t.unsqueeze(-1)
    xt = t * x1 + (1.0 - t) * x0
    target = x1 - x0
    return xt, t.squeeze(), target


def flow_matching_loss(pred_velocity: Tensor, target_velocity: Tensor) -> Tensor:
    """L = ||u_Θ(xt,t,c) - (x1 - x0)||² (Eq. 5)."""
    return F.mse_loss(pred_velocity, target_velocity)
