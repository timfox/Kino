"""Augmented Lagrangian unrolling updates — Sec. 3.2 (CPU stubs)."""

from __future__ import annotations

import math
from typing import Any


def soft_threshold(values: list[float], tau: float) -> list[float]:
    """ℓ1 proximal / soft-threshold operator."""
    return [math.copysign(max(abs(v) - tau, 0.0), v) for v in values]


def update_u(
    h_conv_x: float,
    gamma: float,
    y: float,
    e: float,
    *,
    lambda1: float = 1.0,
) -> float:
    """Ut+1 — Eq. 5 / 13."""
    return (lambda1 * h_conv_x + gamma + y - e) / (1.0 + lambda1)


def update_x_fft_stub(
    h_adj_u: float,
    gamma: float,
    omega: float,
    z: float,
    *,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    h_energy: float = 1.0,
) -> float:
    """Xt+1 — Eq. 6 / 22 (1-D stub of FFT division)."""
    num = h_adj_u * (-gamma + lambda1 * h_adj_u) - omega + lambda2 * z
    den = lambda1 * h_energy + lambda2
    return num / max(den, 1e-8)


def update_e(
    u: float,
    y: float,
    delta: float,
    p: float,
    *,
    lambda3: float = 1.0,
) -> float:
    """Et+1 — Eq. 7 soft-threshold on error term."""
    phi = lambda3 + 1.0
    n = delta + u - y - lambda3 * p
    return soft_threshold([-n / phi], lambda3 / phi)[0]


def cnn_denoise_stub(x: float, scale: float = 1.0) -> float:
    """Zt+1 / Pt+1 ResUNet surrogate — Eq. 8."""
    return x / scale


def alm_multipliers(
    gamma: float,
    omega: float,
    delta: float,
    *,
    h_conv_x: float,
    u: float,
    x: float,
    z: float,
    e: float,
    p: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
) -> tuple[float, float, float]:
    """Γ, Ω, Δ updates — Eq. 9."""
    gamma_n = gamma + lambda1 * (h_conv_x - u)
    omega_n = omega + lambda2 * (x - z)
    delta_n = delta + lambda3 * (e - p)
    return gamma_n, omega_n, delta_n


def unrolling_block_card() -> dict[str, Any]:
    return {
        "variables": ["U", "X", "E", "Z", "P"],
        "multipliers": ["Γ", "Ω", "Δ"],
        "closed_form": ["U", "X", "E"],
        "cnn_refine": ["Z via Dϕ ResUNet", "P via Df ResUNet"],
        "error_aware": "sparse E corrects kernel estimation errors (Eq. 2–3)",
        "depth_K": 10,
    }
