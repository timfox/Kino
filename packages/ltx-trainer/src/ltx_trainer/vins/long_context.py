"""Long-token-sequence generalization: attention + RoPE rescaling (Sec. 4.1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def resolution_ratio(n_uhr: int, n_nhr: int) -> float:
    """sqrt(N_UHR / N_NHR) used for τ and RoPE base scaling."""
    if n_nhr <= 0:
        raise ValueError("n_nhr must be positive")
    return math.sqrt(float(n_uhr) / float(n_nhr))


def attention_temperature(n_uhr: int, n_nhr: int) -> float:
    """τ = log(sqrt(N_UHR / N_NHR)) per paper Sec. 4.1."""
    return math.log(resolution_ratio(n_uhr, n_nhr))


def rescaled_attention_weights(
    q: Tensor,
    k: Tensor,
    *,
    tau: float,
    dim: int = -1,
) -> Tensor:
    """Apply resolution-aware temperature to attention logits (Eq. 4)."""
    scale = (q.shape[-1] ** -0.5) * tau
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.softmax(logits, dim=dim)


def scaled_rope_base(base: float, n_uhr: int, n_nhr: int) -> float:
    """b' = b * sqrt(N_UHR / N_NHR) (NTK-style, Sec. 4.1)."""
    return base * resolution_ratio(n_uhr, n_nhr)


def rope_theta(dim_index: int, position: int, *, base: float, dim: int) -> float:
    """θ_i = b^{-2i/d} for RoPE (Eq. 3)."""
    return float(base ** (-2.0 * dim_index / dim))


def token_count_from_resolution(height: int, width: int, patch: int = 16) -> int:
    """Approximate DiT token count for square-ish latents."""
    return max(1, (height // patch) * (width // patch))
