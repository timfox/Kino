"""LoRA swapping, interpolation, and composition (Eq. 7)."""

from __future__ import annotations

import numpy as np


def compose_lora_updates(
    updates: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """ΔW(w) = Σ_k w_k ΔW_k (Eq. 7)."""
    if not updates:
        raise ValueError("updates must be non-empty")
    out = np.zeros_like(next(iter(updates.values())))
    for name, delta in updates.items():
        out = out + weights.get(name, 0.0) * delta
    return out


def interpolate_opposite(fast: np.ndarray, slow: np.ndarray, alpha: float) -> np.ndarray:
    """Same-axis blend: α·fast + (1-α)·slow."""
    return alpha * fast + (1.0 - alpha) * slow


def compose_axes(
    speed_delta: np.ndarray,
    pitch_delta: np.ndarray,
    *,
    w_speed: float = 0.5,
    w_pitch: float = 0.5,
) -> np.ndarray:
    """Multi-axis composition at default w=0.5 operating point."""
    return w_speed * speed_delta + w_pitch * pitch_delta
