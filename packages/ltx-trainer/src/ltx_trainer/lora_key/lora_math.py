"""LoRA adaptation and linear superposition — Eqs. 2–5, 14."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def lora_delta(b: Tensor, a: Tensor) -> Tensor:
    """∆W = BA with B [d,r], A [r,k]."""
    return b @ a


def lora_forward(h_in: Tensor, w0: Tensor, b: Tensor, a: Tensor, *, alpha: float = 1.0) -> Tensor:
    """h_out = (W0 + α BA) h_in."""
    return nn.functional.linear(h_in, w0) + alpha * nn.functional.linear(nn.functional.linear(h_in, a), b)


def merge_loras(
    w0: Tensor,
    style_b: Tensor,
    style_a: Tensor,
    key_b: Tensor,
    key_a: Tensor,
    *,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> Tensor:
    """Θ_deploy = Θ0 + α∆Θ_s + γ∆Θ_k — training-free superposition (Eq. 14)."""
    return w0 + alpha * lora_delta(style_b, style_a) + gamma * lora_delta(key_b, key_a)


def multi_lora_merge(w0: Tensor, adapters: list[tuple[Tensor, Tensor, float]]) -> Tensor:
    """Θ_merge = Θ0 + Σ α_i ∆Θ_i (Eq. 5)."""
    out = w0.clone()
    for b, a, scale in adapters:
        out = out + scale * lora_delta(b, a)
    return out
