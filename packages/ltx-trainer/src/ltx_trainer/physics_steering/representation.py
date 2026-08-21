"""Mean-pooled patch representations (Eq. 1)."""

from __future__ import annotations

from torch import Tensor


def mean_pool_hidden(H: Tensor) -> Tensor:
    """``f_l(x) = (1/N) Σ_i H_l(x)_i`` — ``H`` is (N, D) or (B, N, D)."""
    if H.ndim == 2:
        return H.mean(dim=0)
    if H.ndim == 3:
        return H.mean(dim=1)
    raise ValueError("H must be (N, D) or (B, N, D)")


def batch_mean_pool(H: Tensor) -> Tensor:
    """(B, N, D) → (B, D)."""
    if H.ndim != 3:
        raise ValueError("batch_mean_pool expects (B, N, D)")
    return H.mean(dim=1)
