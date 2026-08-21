"""Toy MM helpers for cSTMM smoke tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cstmm.density import log_density_unnormalized
from ltx_trainer.cstmm.geometry import normalize_observation
from ltx_trainer.cstmm.mm import responsibilities, update_weights


def random_unit_vectors(n: int, m: int, rng: np.random.Generator) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _ in range(n):
        y = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        z = normalize_observation(y)
        if z is not None:
            out.append(z)
    return out


def component_log_likelihood(
    vecs: list[np.ndarray],
    matrix: np.ndarray,
    *,
    nu: float,
    m_channels: int,
) -> float:
    if not vecs:
        return 0.0
    return float(
        np.mean([log_density_unnormalized(z, matrix, nu, m_channels) for z in vecs])
    )


def toy_mm_step(
    z_list: list[np.ndarray],
    weights: np.ndarray,
    matrices: list[np.ndarray],
    nu: float,
    m_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    gamma = responsibilities(z_list, weights, matrices, nu, m_channels)
    w = update_weights(gamma)
    return gamma, w


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vecs = random_unit_vectors(4, 3, rng)
    m = len(vecs[0])
    weights = np.array([1.0 / len(vecs)] * len(vecs))
    from ltx_trainer.cstmm.mm import build_canonical_a

    matrices = [
        build_canonical_a(np.eye(m), np.array([0.0] + [-0.05] * (m - 1), dtype=np.float64))
        for _ in vecs
    ]
    gamma, w_new = toy_mm_step(vecs, np.array([1.0 / len(vecs)] * len(vecs)), matrices, nu=5.0, m_channels=m)
    ll = component_log_likelihood(vecs, matrices[0], nu=5.0, m_channels=m)
    return {
        "num_sources": len(vecs),
        "gamma_sum": round(float(gamma.sum()), 4),
        "weights_sum": round(float(w_new.sum()), 4),
        "log_likelihood": round(ll, 4),
    }
