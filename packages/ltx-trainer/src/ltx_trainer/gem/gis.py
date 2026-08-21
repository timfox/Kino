"""Geometric Influence Score (GIS) for cluster representatives."""

from __future__ import annotations

import numpy as np

from ltx_trainer.gem.vmf import log_vmf_unnormalized


def cluster_local_density(
    x: np.ndarray,
    cluster_ids: np.ndarray,
    k: int,
    cluster_index: int,
    neighbor_m: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Mean cosine to top-M neighbors within the same hard cluster."""
    mask = cluster_ids == cluster_index
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros(len(x), dtype=np.float64)
    rho = np.zeros(len(x), dtype=np.float64)
    sub = x[idx]
    sims = sub @ sub.T
    np.fill_diagonal(sims, -1.0)
    m = min(neighbor_m, max(1, len(idx) - 1))
    for local_i, global_i in enumerate(idx):
        row = sims[local_i]
        top = np.partition(row, -m)[-m:]
        rho[global_i] = float(np.mean(top))
    return np.clip(rho, eps, None)


def gis_score(
    x: np.ndarray,
    gamma: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    cluster_index: int,
    *,
    beta: float = 0.5,
    neighbor_m: int = 8,
    eps: float = 1e-8,
) -> np.ndarray:
    """GIS_k(x_i) = log γ_ik + log f_ik + β log ρ_k (Eq. 21)."""
    hard = gamma.argmax(axis=1)
    rho = cluster_local_density(x, hard, gamma.shape[1], cluster_index, neighbor_m, eps=eps)
    g = np.clip(gamma[:, cluster_index], eps, 1.0)
    log_f = log_vmf_unnormalized(x, mu[cluster_index], kappa[cluster_index])
    scores = np.log(g) + log_f + beta * np.log(np.clip(rho, eps, None))
    scores[hard != cluster_index] = -np.inf
    return scores


def top_gis_indices(
    x: np.ndarray,
    gamma: np.ndarray,
    mu: np.ndarray,
    kappa: np.ndarray,
    cluster_index: int,
    top_m: int,
    **kwargs: float | int,
) -> list[int]:
    s = gis_score(x, gamma, mu, kappa, cluster_index, **kwargs)  # type: ignore[arg-type]
    hard = gamma.argmax(axis=1)
    pool = np.where(hard == cluster_index)[0]
    if len(pool) == 0:
        return []
    order = pool[np.argsort(-s[pool])]
    return order[:top_m].tolist()
