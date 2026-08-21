"""Stochastic SLEκ Loewner dynamics (Eq. 3) via Euler–Maruyama."""

from __future__ import annotations

import numpy as np


def euler_maruyama_sle(
    kappa: float,
    z0: complex,
    *,
    t_end: float = 1.0,
    dt: float = 0.01,
    seed: int = 0,
    brownian_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler update g_{n+1} = g_n + Δt · 2/(g_n − √κ W_n)."""
    n = max(1, int(t_end / dt))
    times = np.linspace(0.0, t_end, n + 1)
    g = np.empty(n + 1, dtype=np.complex128)
    g[0] = z0

    if brownian_path is not None:
        w = np.asarray(brownian_path, dtype=np.float64)
        if w.shape[0] != n + 1:
            raise ValueError("brownian_path length must match time grid")
    else:
        rng = np.random.default_rng(seed)
        dw = rng.normal(scale=np.sqrt(dt), size=n)
        w = np.concatenate([[0.0], np.cumsum(dw)])

    sqrt_k = np.sqrt(max(kappa, 0.0))
    for i in range(n):
        denom = g[i] - sqrt_k * w[i]
        if abs(denom) < 1e-8:
            denom = denom + 1e-8j
        g[i + 1] = g[i] + dt * (2.0 / denom)
    return times, g


def sle_trace_boundary_proxy(g: np.ndarray) -> np.ndarray:
    """Proxy for lim_{y→0+} g⁻¹_t(iy): use imaginary part decay along trajectory."""
    return np.stack([g.real, g.imag, np.abs(g)], axis=1)


def generate_sle_dataset(
    n: int,
    *,
    kappa_range: tuple[float, float] = (0.0, 8.0),
    seed: int = 0,
    shared_brownian: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Monte Carlo ensemble of SLE trajectories and κ labels."""
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    kappas = rng.uniform(kappa_range[0], kappa_range[1], size=n)
    shared_w: np.ndarray | None = None
    if shared_brownian:
        dt = 0.01
        n = max(1, int(1.0 / dt))
        rng_bm = np.random.default_rng(seed)
        dw = rng_bm.normal(scale=np.sqrt(dt), size=n)
        shared_w = np.concatenate([[0.0], np.cumsum(dw)])

    for i in range(n):
        kappa = float(kappas[i])
        x0, y0 = rng.uniform(0.0, 10.0, size=2)
        z0 = complex(x0, y0)
        path = shared_w if shared_brownian else None
        _, g = euler_maruyama_sle(kappa, z0, seed=seed + i + 1, brownian_path=path)
        flat = np.concatenate([g.real, g.imag])
        features.append(flat)
    return features, kappas
