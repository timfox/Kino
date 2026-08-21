"""DSTA adapter forward pass (Sec. IV-C, Eq. 1)."""

from __future__ import annotations

import math


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def dsta_forward_scalar(
    x: float,
    *,
    w_down: float,
    w_mid: float,
    w_up: float,
    dst_out: float,
    beta: float = 1.0,
) -> float:
    r"""
    Toy scalar DSTA: x' = β·W_up·(W_mid·DST(σ(W_down·x)) + σ(W_down·x)) + x (Eq. 1).
    """
    x_bar = gelu(w_down * x)
    x_hat = w_mid * dst_out + x_bar
    return beta * w_up * x_hat + x
