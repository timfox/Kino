"""Flow-matching training loss (Sec. 3.2, Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def flow_matching_loss(
    velocity_pred: Tensor,
    x0: Tensor,
    epsilon: Tensor,
) -> Tensor:
    """L_FM = ||vθ(x_t, t) - (x0 - ε)||² (Eq. 1)."""
    target = x0 - epsilon
    return torch.mean((velocity_pred - target) ** 2)


def flow_matching_step_targets(
    x0: Tensor,
    t: float | Tensor,
    epsilon: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, epsilon, target_velocity) for one FM step."""
    if epsilon is None:
        epsilon = torch.randn_like(x0)
    if not isinstance(t, Tensor):
        t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    x_t = (1.0 - t) * x0 + t * epsilon
    return x_t, epsilon, x0 - epsilon
