"""PLS-SVD decomposition and similarity dissection (Eq. 1–9)."""

from __future__ import annotations

from typing import Any

import numpy as np


def center_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (centered, mean) for N×C embedding matrix."""
    x = np.asarray(embeddings, dtype=np.float64)
    mean = x.mean(axis=0)
    return x - mean, mean


def pls_svd(text: np.ndarray, audio: np.ndarray) -> dict[str, np.ndarray]:
    """Eq. (2): M = T^T A = U Σ V^T on centered paired embeddings."""
    t, t_mean = center_embeddings(text)
    a, a_mean = center_embeddings(audio)
    m = t.T @ a
    u, sigma, vt = np.linalg.svd(m, full_matrices=True)
    v = vt.T
    return {
        "T": t,
        "A": a,
        "t_mean": t_mean,
        "a_mean": a_mean,
        "U": u,
        "V": v,
        "Sigma": np.diag(sigma),
        "sigma": sigma,
        "M": m,
    }


def project_coefficients(
    centered: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Projection coefficients ˆ = X @ directions (N×C)."""
    return centered @ directions


def covariance_decomposition(
    t_coef: np.ndarray,
    a_coef: np.ndarray,
) -> dict[str, float]:
    """Eq. (4): normalized covariance = Corr * sqrt(Var_t) * sqrt(Var_a)."""
    n = t_coef.size
    cov = float(t_coef @ a_coef / max(n, 1))
    var_t = float(t_coef @ t_coef / max(n, 1))
    var_a = float(a_coef @ a_coef / max(n, 1))
    std_t, std_a = np.sqrt(max(var_t, 1e-12)), np.sqrt(max(var_a, 1e-12))
    corr = cov / (std_t * std_a + 1e-12)
    corr = float(np.clip(corr, -1.0, 1.0))
    return {
        "cov": cov,
        "std_t": std_t,
        "std_a": std_a,
        "corr": corr,
    }


def similarity_direct_cross(
    t_hat: np.ndarray,
    a_hat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    head: int | None = None,
) -> dict[str, float]:
    """Eq. (6–8): direct vs cross effect contributions for one pair."""
    c = u.shape[0]
    k = c if head is None else min(head, c)
    direct = 0.0
    cross = 0.0
    for j in range(k):
        direct += float(t_hat[j] * a_hat[j] * (u[:, j] @ v[:, j]))
    for l in range(c):
        for m in range(c):
            if l == m:
                continue
            cross += float(t_hat[l] * a_hat[m] * (u[:, l] @ v[:, m]))
    return {"direct": direct, "cross": cross, "direct_head": direct if head is None else direct}


def net_useful_contribution(sigma_jj: float, uv_align: float) -> float:
    """Eq. (9): Σ_jj × (u_j · v_j)."""
    return float(sigma_jj * uv_align)


def reconstruct_from_projections(
    mean: np.ndarray,
    coef: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Eq. (3): x = mean + Σ_j coef_j u_j."""
    return mean + directions @ coef
