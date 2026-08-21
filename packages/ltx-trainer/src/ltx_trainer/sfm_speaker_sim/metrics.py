"""Human–model speaker-similarity correspondence metrics (Sec. 3.3)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def upper_triangle_values(m: np.ndarray) -> np.ndarray:
    """Deduplicate undirected speaker pairs (exclude diagonal)."""
    n = m.shape[0]
    idx = np.triu_indices(n, k=1)
    return m[idx].astype(np.float64)


def pearson_lcc(human: np.ndarray, model: np.ndarray) -> float:
    h = upper_triangle_values(human)
    m = upper_triangle_values(model)
    if h.size < 2:
        return 0.0
    r, _ = stats.pearsonr(h, m)
    return float(r)


def spearman_srcc(human: np.ndarray, model: np.ndarray) -> float:
    h = upper_triangle_values(human)
    m = upper_triangle_values(model)
    if h.size < 2:
        return 0.0
    r, _ = stats.spearmanr(h, m)
    return float(r)


def frobenius_distance(human: np.ndarray, model: np.ndarray) -> float:
    diff = human.astype(np.float64) - model.astype(np.float64)
    return float(np.linalg.norm(diff, ord="fro"))


def normalized_laplacian_eigenvalues(adj: np.ndarray, k: int) -> np.ndarray:
    """k smallest non-zero eigenvalues of the normalized graph Laplacian."""
    a = adj.astype(np.float64)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(axis=1)
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    lap = np.eye(a.shape[0]) - d_inv_sqrt @ a @ d_inv_sqrt
    vals = np.sort(np.linalg.eigvalsh(lap))
    nz = vals[vals > 1e-8]
    if nz.size == 0:
        return np.zeros(k, dtype=np.float64)
    take = min(k, nz.size)
    out = np.zeros(k, dtype=np.float64)
    out[:take] = nz[:take]
    return out


def spectral_distance(human: np.ndarray, model: np.ndarray, k: int = 10) -> float:
    eh = normalized_laplacian_eigenvalues(human, k)
    em = normalized_laplacian_eigenvalues(model, k)
    return float(np.linalg.norm(eh - em, ord=2))


def correspondence_bundle(
    human: np.ndarray,
    model: np.ndarray,
    *,
    spectral_k: int = 10,
) -> dict[str, float]:
    return {
        "lcc": pearson_lcc(human, model),
        "srcc": spearman_srcc(human, model),
        "frobenius": frobenius_distance(human, model),
        "spectral": spectral_distance(human, model, k=spectral_k),
    }
