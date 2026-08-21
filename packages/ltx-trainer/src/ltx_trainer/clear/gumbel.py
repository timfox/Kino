"""Differentiable layer selection via Gumbel–Softmax (Sec. 3.2, Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def gumbel_softmax(
    logits: Tensor,
    *,
    tau: float = 1.0,
    hard: bool = False,
) -> Tensor:
    """Sample soft layer distribution p over L layers."""
    if logits.dim() != 1:
        raise ValueError("logits must be shape (L,)")
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = torch.softmax((logits + g) / tau, dim=0)
    if hard:
        idx = int(torch.argmax(y).item())
        hard_vec = torch.zeros_like(y)
        hard_vec[idx] = 1.0
        return hard_vec
    return y


def select_layer_index(alpha: Tensor) -> int:
    """Inference: l* = argmax_l alpha_l."""
    return int(torch.argmax(alpha).item())
