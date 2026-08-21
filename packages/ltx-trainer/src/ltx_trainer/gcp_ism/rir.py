"""Forward / inverse RIR constructions (Eqs. 16, 18, 20)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def lanczos(x: Tensor, alpha: int) -> Tensor:
    """Lanczos kernel L(x, α) for inverse construction (Eq. 20)."""
    if alpha <= 0:
        return torch.zeros_like(x)
    out = torch.sinc(x) * torch.sinc(x / alpha)
    return torch.where(x.abs() <= alpha, out, torch.zeros_like(out))


def forward_rir_sample(
    vol_q: Tensor,
    ki: float,
    k_ip1: float,
    n_dims: int,
    *,
    lambda_scale: int = 1,
    k_threshold: float = 0.0,
) -> float:
    """Single sample from Eq. (18) style finite difference."""
    denom = max(ki ** (n_dims - 1), 1e-12)
    if ki > k_threshold and lambda_scale > 1:
        qi = lambda_scale * lambda_scale * round(ki * ki)
        qip1 = lambda_scale * lambda_scale * round(k_ip1 * k_ip1)
        dq = float(vol_q[min(qip1, vol_q.numel() - 1)] - vol_q[min(qi, vol_q.numel() - 1)])
    else:
        q_i = int(round(ki * ki))
        q_ip1 = int(round(k_ip1 * k_ip1))
        dq = float(vol_q[min(q_ip1, vol_q.numel() - 1)] - vol_q[min(q_i, vol_q.numel() - 1)])
    return dq / denom


def inverse_rir_stub(length: int, *, alpha: int = 10) -> Tensor:
    """Placeholder inverse RIR from Lanczos-weighted differences (smoke)."""
    q_len = max(length, 8)
    vol = torch.cumsum(torch.ones(q_len), dim=0)
    h = torch.zeros(length)
    for i in range(length):
        acc = 0.0
        for q in range(q_len - 1):
            dq = float(vol[q + 1] - vol[q])
            dist = math.sqrt(max(q, 0)) / max(length, 1)
            acc += dq * float(lanczos(torch.tensor(i - dist), alpha))
        h[i] = acc / max(i + 1, 1) ** 0.5
    return h
