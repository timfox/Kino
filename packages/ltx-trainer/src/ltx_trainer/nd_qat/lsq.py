"""Learned Step Size Quantization (LSQ) with STE (Sec. 1.2.1, Eq. 1–3)."""

from __future__ import annotations

import torch
from torch import Tensor


class LSQQuantize(torch.autograd.Function):
    """Ternary LSQ: clip(round(w/Δ)*Δ, -Δ, Δ) with STE backward."""

    @staticmethod
    def forward(ctx, weight: Tensor, step_size: Tensor) -> Tensor:
        delta = step_size.abs().clamp(min=1e-8)
        q = torch.round(weight / delta) * delta
        q = torch.clamp(q, -delta, delta)
        ctx.save_for_backward(weight, q, delta)
        return q

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor | None]:
        # STE: ∂L/∂w ≈ ∂L/∂ŵ (Eq. 3)
        return grad_output, None


def lsq_quantize(weight: Tensor, step_size: Tensor) -> Tensor:
    return LSQQuantize.apply(weight, step_size)


def ternary_project(weight: Tensor, step_size: Tensor) -> Tensor:
    """Map to {−1, 0, +1} via sign after LSQ clip."""
    q = lsq_quantize(weight, step_size)
    delta = step_size.abs().clamp(min=1e-8)
    signs = torch.sign(q)
    zero_mask = q.abs() < 0.5 * delta
    out = signs.clone()
    out[zero_mask] = 0
    return out


def quantize_entropy_bits(values: tuple[int, ...] = (-1, 0, 1)) -> float:
    """Equivalent bit-width for ternary alphabet (paper: 1.58 bits)."""
    import math

    n = len(values)
    return math.log2(n)
