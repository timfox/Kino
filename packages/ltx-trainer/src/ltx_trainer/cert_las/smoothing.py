"""Layer-adaptive Gaussian smoothing and certified radius (Cert-LAS §4.2, Theorem 4.9)."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ltx_trainer.cert_las.stats import phi_cdf, phi_inv


def mahalanobis_norm(delta_layers: Mapping[str, np.ndarray], sigmas: Mapping[str, float]) -> float:
    """Definition A.2: ||δ||_{σ_k}."""
    total = 0.0
    for name, delta in delta_layers.items():
        sigma = float(sigmas.get(name, 1.0))
        if sigma <= 0:
            continue
        total += float(np.sum(delta.ravel() ** 2)) / (sigma**2)
    return math.sqrt(total)


def sample_layer_noise(
    sigmas: Mapping[str, float],
    dims: Mapping[str, int],
    *,
    rng: np.random.Generator | None = None,
    scale_k: float = 1.0,
) -> dict[str, np.ndarray]:
    gen = rng or np.random.default_rng()
    return {
        name: gen.standard_normal(dims[name]) * float(sigmas[name]) * scale_k
        for name in sigmas
    }


def apply_noise_to_layers(
    base: Mapping[str, np.ndarray],
    noise: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {k: base[k] + noise[k] for k in base}


def certified_radius_grid(
    ps: list[float],
    *,
    tau: float,
    k: float = 1.0,
    a: float = 0.0,
    b: float = 1.0,
    grid: int = 80,
) -> float:
    """
    Solve Theorem 4.9 / Eq. (6) for maximal ¯R via monotone grid search.

    ps: lower confidence bounds P_{s_j}(θ) at quantile thresholds s_j.
    """
    if not ps:
        return 0.0
    s_vals = [a] + list(ps)
    if len(s_vals) < 2:
        return 0.0

    def lhs(r_bar: float) -> float:
        acc = s_vals[0]
        for j in range(1, len(s_vals)):
            sj, sj1 = s_vals[j], s_vals[j - 1]
            z = phi_cdf(phi_inv(sj) - r_bar / max(k, 1e-9))
            acc += (sj - sj1) * z
        return acc

    lo, hi = 0.0, 4.0
    if lhs(lo) <= tau:
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if lhs(mid) > tau:
            lo = mid
        else:
            hi = mid
    return lo
