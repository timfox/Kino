"""CFG and AdaMaG guidance velocity helpers (toy / reference)."""

from __future__ import annotations

import math
from typing import Sequence


def cfg_velocity(
    v_uncond: Sequence[float],
    v_cond: Sequence[float],
    omega: float,
) -> list[float]:
    """Classifier-free guidance: v_u + ω(v_c - v_u)."""
    return [u + omega * (c - u) for u, c in zip(v_uncond, v_cond, strict=True)]


def omega_schedule(
    t: float,
    omega_ref: float,
    *,
    omega_min: float = 1.0,
    gamma: float = 4.0,
) -> float:
    """Eq. (11): ω(t) = max(ω_min, ω_ref (1-t)^γ)."""
    t = max(0.0, min(1.0, t))
    return max(omega_min, omega_ref * ((1.0 - t) ** gamma))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm2(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _scale(a: Sequence[float], s: float) -> list[float]:
    return [x * s for x in a]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b, strict=True)]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


def score_normal_direction(
    x: Sequence[float],
    v_cond: Sequence[float],
    *,
    a_t: float = 1.0,
) -> list[float]:
    """n_t = a_t x - v_c (Eq. 10, using conditional flow)."""
    return _sub(_scale(x, a_t), v_cond)


def decompose_guidance(
    g: Sequence[float],
    n: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Project g into parallel and orthogonal components w.r.t. n."""
    n2 = _dot(n, n)
    if n2 < 1e-12:
        return list(g), [0.0] * len(g)
    coeff = _dot(g, n) / n2
    g_parallel = _scale(n, coeff)
    g_orth = _sub(g, g_parallel)
    return g_parallel, g_orth


def adamag_guidance_residual(
    g: Sequence[float],
    n: Sequence[float],
    *,
    omega_t: float,
    beta: float = 0.1,
) -> list[float]:
    """Eq. (13): g̃ = ω(t)(g⊥ + β g∥)."""
    g_par, g_orth = decompose_guidance(g, n)
    g_tilde = _add(g_orth, _scale(g_par, beta))
    return _scale(g_tilde, omega_t)


def adamag_velocity(
    v_uncond: Sequence[float],
    v_cond: Sequence[float],
    x: Sequence[float],
    *,
    t: float,
    omega_ref: float,
    beta: float = 0.1,
    gamma: float = 4.0,
    omega_min: float = 1.0,
    a_t: float = 1.0,
) -> list[float]:
    """Full AdaMaG guided velocity: v_u + g̃."""
    g = _sub(v_cond, v_uncond)
    n = score_normal_direction(x, v_cond, a_t=a_t)
    w = omega_schedule(t, omega_ref, omega_min=omega_min, gamma=gamma)
    g_tilde = adamag_guidance_residual(g, n, omega_t=w, beta=beta)
    return _add(v_uncond, g_tilde)
