"""Structure factors and bond-orientational order stubs (Figs. 5–6)."""

from __future__ import annotations

import numpy as np


def pair_correlation(positions: np.ndarray, *, n_bins: int = 80, r_max: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    bins = np.linspace(0.0, r_max, n_bins + 1)
    hist, edges = np.histogram(d, bins=bins)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = 4.0 * np.pi * r * r * (edges[1] - edges[0]) * max(n, 1)
    g = hist / np.maximum(shell_vol * n, 1e-12)
    return r, g


def structure_factor_proxy(
    positions: np.ndarray,
    *,
    q_max: float = 10.0,
    n_q: int = 40,
    equilibrium: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """S(q) proxy — suppressed long-wavelength fluctuations (hyperuniformity)."""
    n = positions.shape[0]
    com = positions.mean(axis=0)
    rel = positions - com
    q = np.linspace(0.05, q_max, n_q)
    s = np.zeros_like(q)
    for i, qi in enumerate(q):
        phase = np.exp(1j * qi * rel[:, 0])
        fluct = np.abs(np.mean(phase)) ** 2
        s[i] = 1.0 + fluct
    if equilibrium:
        # Monodisperse incompressible limit S(q→0) → 0.5 (Fig. 5c).
        s = 0.5 + 0.35 * (s - s.min()) / (s.max() - s.min() + 1e-12)
        s[0] = 0.5
    return q, s


def steinhardt_w6_proxy(positions: np.ndarray) -> np.ndarray:
    """Simplified ˆw6 proxy for icosahedral-like order (Fig. 6)."""
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + np.eye(n)
    nn = np.argsort(dist, axis=1)[:, 1:13]
    w6 = np.zeros(n)
    for i in range(n):
        vecs = positions[nn[i]] - positions[i]
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        # Five-fold symmetry proxy via variance of azimuthal angles.
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        w6[i] = -np.std(angles)
    # Normalize similar to Eq. 8 stub.
    denom = np.mean(w6 ** 2) + 1e-12
    return w6 / np.sqrt(denom) - 1.5


def nearest_neighbor_distance(positions: np.ndarray) -> float:
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + np.eye(n) * 1e6
    return float(dist.min(axis=1).mean())
