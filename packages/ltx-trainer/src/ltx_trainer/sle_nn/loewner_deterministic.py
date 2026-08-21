"""Deterministic Loewner ODE with ξ(t)=c√t (Eq. 1–2)."""

from __future__ import annotations

from typing import Any

import numpy as np


def loewner_exponent_a(c: float) -> float:
    """a = 1/2 − c / (2√(16+c²)) from Eq. (2)."""
    c = float(c)
    return 0.5 - 0.5 * c / np.sqrt(16.0 + c * c)


def loewner_alpha(c: float, t: float) -> complex:
    """α = 2√t √(a/(1−a)) in the inverse map."""
    a = loewner_exponent_a(c)
    if abs(1.0 - a) < 1e-12:
        return 0.0 + 0.0j
    return 2.0 * np.sqrt(t) * np.sqrt(a / (1.0 - a))


def inverse_loewner_map(w: complex, t: float, c: float) -> complex:
    """g⁻¹_t(w) for ξ(t)=c√t (Eq. 2)."""
    if t <= 0.0:
        return w
    a = loewner_exponent_a(c)
    alpha = loewner_alpha(c, t)
    left = (w + alpha) ** (1.0 - a)
    right = (w - alpha) ** a
    return left * right


def forward_loewner_newton(
    z0: complex,
    t: float,
    c: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 40,
) -> complex:
    """Invert g⁻¹_t(w)=z0 via Newton–Raphson (Sec. 2.4)."""
    if t <= 0.0:
        return z0
    a = loewner_exponent_a(c)
    alpha = loewner_alpha(c, t)
    w = complex(z0)

    for _ in range(max_iter):
        left = w + alpha
        right = w - alpha
        fn = left ** (1.0 - a) * right**a - z0
        if abs(fn) < tol:
            break
        dleft = (1.0 - a) * left ** (-a) * right**a
        dright = a * left ** (1.0 - a) * right ** (a - 1.0)
        deriv = dleft + dright
        if abs(deriv) < 1e-14:
            break
        w -= fn / deriv
    return w


def sample_deterministic_trajectory(
    c: float,
    z0: complex,
    *,
    t_start: float = 0.1,
    t_end: float = 1.0,
    n_steps: int = 100,
) -> np.ndarray:
    """Forward map t ↦ g_t(z0) on a uniform time grid."""
    times = np.linspace(t_start, t_end, n_steps)
    traj = np.empty(n_steps, dtype=np.complex128)
    for i, t in enumerate(times):
        traj[i] = forward_loewner_newton(z0, float(t), c)
    return traj


def trajectory_features(traj: np.ndarray) -> np.ndarray:
    """Flatten real/imag trajectory for NN input."""
    return np.concatenate([traj.real, traj.imag]).astype(np.float64)
