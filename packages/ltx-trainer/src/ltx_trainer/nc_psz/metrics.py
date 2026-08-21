"""PSZ isolation and stability metrics (toy) for NC-PSZ (arXiv:2605.21891)."""

from __future__ import annotations

import numpy as np


def isolation_ratio_db(etar: float, eleak_or_eint: float, eps: float = 1e-12) -> float:
    """10*log10(E_tar / (E_other + eps)) for IZI/IPI-style metrics."""
    return float(10.0 * np.log10(max(etar, eps) / (max(eleak_or_eint, eps) + eps)))


def neighborhood_median(values: np.ndarray) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def neighborhood_cvar10(values: np.ndarray) -> float:
    """Empirical CVaR10 (Eq. 24)."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = v.size
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.1 * n)))
    return float(np.mean(v[:k]))


def neighborhood_min(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    return float(np.min(v)) if v.size else float("nan")


def stability_variation_rates(
    values: np.ndarray,
    coords: np.ndarray,
    edges: list[tuple[int, int]],
) -> tuple[float, float]:
    """σ_mean and σ_rms in dB/m (Eq. 25–26)."""
    vals = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    sigmas: list[float] = []
    for i, j in edges:
        dij = float(np.linalg.norm(coords[i] - coords[j]))
        if dij <= 0:
            continue
        sigmas.append(abs(vals[i] - vals[j]) / dij)
    if not sigmas:
        return 0.0, 0.0
    arr = np.asarray(sigmas, dtype=np.float64)
    return float(arr.mean()), float(np.sqrt(np.mean(arr**2)))


def improvement_quality_pct(q_nc: float, q_base: float, eps: float = 1e-12) -> float:
    """Eq. 27: higher is better."""
    return 100.0 * (q_nc - q_base) / (abs(q_base) + eps)


def improvement_stability_pct(s_base: float, s_nc: float, eps: float = 1e-12) -> float:
    """Eq. 28: lower is better."""
    return 100.0 * (s_base - s_nc) / (s_base + eps)
