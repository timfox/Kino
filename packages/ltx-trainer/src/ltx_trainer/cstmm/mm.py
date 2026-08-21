"""Generalized MM updates — Eqs. (4), (14)–(15), (24), (27)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cstmm.density import log_density_unnormalized
from ltx_trainer.cstmm.geometry import quadratic_form


def responsibilities(
    z_list: list[np.ndarray],
    weights: np.ndarray,
    matrices: list[np.ndarray],
    nu: float,
    m_channels: int,
) -> np.ndarray:
    """Soft masks γ^{(n)}_{tf} — Eq. (4). Shape (T, N)."""
    t_len = len(z_list)
    n_src = len(matrices)
    log_resp = np.zeros((t_len, n_src), dtype=np.float64)
    for t, z in enumerate(z_list):
        for n in range(n_src):
            log_resp[t, n] = np.log(max(weights[n], 1e-12)) + log_density_unnormalized(
                z, matrices[n], nu, m_channels
            )
        log_resp[t] -= np.max(log_resp[t])
        exp_r = np.exp(log_resp[t])
        s = exp_r.sum()
        if s > 0:
            log_resp[t] = exp_r / s
        else:
            log_resp[t] = 1.0 / n_src
    return log_resp


def phi_auxiliary(z_list: list[np.ndarray], matrices: list[np.ndarray], gamma: np.ndarray, nu: float) -> np.ndarray:
    """ϕ^{(n)}_{tf} ← 2/ν z^H A^{(n)} z — Eq. (14)."""
    t_len, n_src = gamma.shape
    phi = np.zeros_like(gamma)
    for t, z in enumerate(z_list):
        for n in range(n_src):
            phi[t, n] = (2.0 / nu) * quadratic_form(z, matrices[n])
    return phi


def update_weights(gamma: np.ndarray) -> np.ndarray:
    """w^{(n)}_f ← mean_t γ^{(n)}_{tf} — Eq. (15)."""
    return gamma.mean(axis=0)


def scatter_matrix(
    z_list: list[np.ndarray],
    gamma: np.ndarray,
    phi: np.ndarray,
    component: int,
    nu: float,
) -> np.ndarray:
    """S for eigenvector / HCA updates — Eq. (25) style."""
    m = len(z_list[0])
    s = np.zeros((m, m), dtype=np.complex128)
    for t, z in enumerate(z_list):
        g = gamma[t, component]
        if g < 1e-12:
            continue
        denom = max(1.0 - phi[t, component], 1e-8)
        s += ((nu + m) / nu) * g / denom * np.outer(z, z.conj())
    return s


def hca_eigenvalues(s: np.ndarray, gamma_sum: float, nu: float) -> np.ndarray:
    """λ^{HCA}_j = -G/σ_j for j=2..M — Eq. (24)."""
    sigma = np.linalg.eigvalsh(np.asarray(s, dtype=np.complex128))
    sigma = np.sort(sigma)[::-1]
    m = len(sigma)
    lambdas = np.zeros(m, dtype=np.float64)
    for j in range(1, m):
        if sigma[j] > 1e-12:
            lambdas[j] = -gamma_sum / sigma[j]
    return lambdas


def hca_concentration(s: np.ndarray, gamma_sum: float, m_channels: int) -> float:
    """κ^{HCA} — Eq. (27) for Watson-constrained (rank-one) case."""
    sigma = np.linalg.eigvalsh(np.asarray(s, dtype=np.complex128))
    sigma = np.sort(sigma)[::-1]
    tail = float(np.sum(sigma[1:]))
    if tail < 1e-12:
        return 0.0
    return gamma_sum * (m_channels - 1) / tail


def build_canonical_a(eigenvectors: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """Hermitian A with λ_max = 0 — canonical representation §3.1."""
    d = np.diag(eigenvalues.astype(np.complex128))
    a = eigenvectors @ d @ eigenvectors.conj().T
    a = (a + a.conj().T) / 2
    shift = float(np.max(np.linalg.eigvalsh(a)))
    a = a - shift * np.eye(a.shape[0], dtype=np.complex128)
    return a
