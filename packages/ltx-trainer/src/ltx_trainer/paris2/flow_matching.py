"""Flow-matching velocity objective for video experts (Sec. 3.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def _broadcast_time(t: Tensor, x: Tensor) -> Tensor:
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t


def sample_linear_path(x0: Tensor, t: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """x_t = (1-t) x0 + t ε; target velocity v = ε - x0."""
    if noise is None:
        noise = torch.randn_like(x0)
    tb = _broadcast_time(t, x0)
    x_t = (1.0 - tb) * x0 + tb * noise
    v_target = noise - x0
    return x_t, v_target


def flow_matching_loss(v_pred: Tensor, v_target: Tensor) -> Tensor:
    """Mean squared error on velocity field."""
    return torch.nn.functional.mse_loss(v_pred, v_target)


class ExpertVelocityStub(torch.nn.Module):
    """Lightweight stand-in for 11B MM-DiT expert."""

    def __init__(self, channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(channels, hidden, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(hidden, channels, kernel_size=3, padding=1),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        tb = _broadcast_time(t, x_t)
        return self.net(x_t) * (1.0 + 0.1 * tb)
