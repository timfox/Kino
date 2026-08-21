"""Toy RobustSpeechFlow losses (Eq. 1–3, 6–7; Algorithm 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def linear_path(x: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    """Eq. (1): x_t = (1 − t) ε + t x."""
    if x.shape != eps.shape:
        raise ValueError("x and eps must match shape")
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * eps + t * x


def velocity(x: Tensor, eps: Tensor) -> Tensor:
    """Target velocity: v(x, ε) = x − ε (Sec. 3.1)."""
    if x.shape != eps.shape:
        raise ValueError("x and eps must match shape")
    return x - eps


def mse(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError("tensors must match shape")
    return torch.mean((a - b) ** 2)


def robustspeechflow_objective(
    u_pred: Tensor,
    *,
    x: Tensor,
    eps: Tensor,
    x_rand: Tensor,
    x_aug: Tensor,
    lambda_rand: float = 0.2,
    lambda_aug: float = 0.2,
) -> dict[str, Tensor]:
    """Algorithm 1 / Eq. (7): L = Lpos − λrand Lrand − λaug Laug."""
    v_pos = velocity(x, eps)
    v_rand = velocity(x_rand, eps)
    v_aug = velocity(x_aug, eps)

    l_pos = mse(u_pred, v_pos)
    l_rand = mse(u_pred, v_rand)
    l_aug = mse(u_pred, v_aug)

    l_total = l_pos - float(lambda_rand) * l_rand - float(lambda_aug) * l_aug
    return {
        "Lpos": l_pos,
        "Lrand": l_rand,
        "Laug": l_aug,
        "Ltotal": l_total,
    }

