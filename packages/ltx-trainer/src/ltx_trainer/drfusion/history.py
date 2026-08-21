"""Stabilized History Guidance + Soft Temporal Anchoring (Eq. 6, Sec. 3.3)."""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor


class HistoryMode(str, Enum):
    """Parallel history configurations H(0), H(1), H(2)."""

    BASELINE = "H0"  # pure noise — temporal-unconditional
    STABILIZED = "H1"  # minimal noise — low-pass motion reference
    CONTEXT_SUPPRESS = "H2"  # only previous frame retained


def noise_modulate(z: Tensor, sigma: float, *, generator: torch.Generator | None = None) -> Tensor:
    """N_λ(z) = α_λ z + σ_λ ε (Sec. 3.3.1)."""
    if sigma <= 0:
        return z
    eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
    return z + sigma * eps


def build_history_window(
    history: Tensor,
    mode: HistoryMode,
    *,
    stabilize_sigma: float = 0.02,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply spectral-temporal modulation to historical latents (B, T_hist, C, H, W)."""
    if mode == HistoryMode.BASELINE:
        return torch.randn_like(history)
    if mode == HistoryMode.STABILIZED:
        return noise_modulate(history, stabilize_sigma, generator=generator)
    # H(2): keep only most recent temporal frame
    out = torch.zeros_like(history)
    if history.dim() == 5:
        out[:, :, -1:] = history[:, :, -1:]
    else:
        out[:, -1:] = history[:, -1:]
    return out


def stabilized_history_guidance(
    v_h0: Tensor,
    v_h1: Tensor,
    v_h2: Tensor,
    guidance_scale: float = 2.0,
) -> Tensor:
    """Eq. (6): v_guided = v(H0) + s · (v(H1) − v(H2))."""
    return v_h0 + guidance_scale * (v_h1 - v_h2)
