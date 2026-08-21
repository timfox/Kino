"""Optimizers for hybrid SPIM-EP: SGD (λ) and Binary Optimizer (ξ)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sgd_step(
    params: NDArray[np.floating],
    grad: NDArray[np.floating],
    lr: float,
    *,
    l2: float = 0.0,
) -> NDArray[np.floating]:
    return params - lr * (grad + l2 * params)


def bop_step(
    xi: NDArray[np.floating],
    grad_ema: NDArray[np.floating],
    tau: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Flip binary ξ_{k,i} when EMA gradient exceeds threshold τ (BOP [47,48])."""
    out = xi.copy()
    flip = np.abs(grad_ema) > tau
    out[flip] = -out[flip]
    return out, grad_ema
