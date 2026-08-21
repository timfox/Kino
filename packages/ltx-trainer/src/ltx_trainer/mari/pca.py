"""In-field PCA subspace for off-subspace regularization (Eq. 15, 30)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCASubspace:
    B: np.ndarray  # (d, k) orthonormal columns

    def project_in(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        return self.B @ (self.B.T @ v)

    def project_off(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        return v - self.project_in(v)

    def off_subspace_penalty(self, delta: np.ndarray) -> float:
        off = self.project_off(delta)
        return float(np.dot(off, off))


def fit_pca(hiddens: np.ndarray, k: int) -> PCASubspace:
    """Fit rank-k PCA on rows of hiddens (n, d)."""
    h = np.asarray(hiddens, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("hiddens must be (n, d)")
    n, d = h.shape
    k = min(k, n, d)
    mean = h.mean(axis=0, keepdims=True)
    centered = h - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    B = vt[:k].T
    q, _ = np.linalg.qr(B, mode="reduced")
    return PCASubspace(B=q[:, :k])
