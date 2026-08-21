"""Evolution kernels from PDE discretization — Sec. 2.1, 6."""

from __future__ import annotations

import math

import numpy as np

from ltx_trainer.dynamic_gp.basis import gram_matrix


def heat_kernel(x: np.ndarray, s: np.ndarray, *, alpha: float, dt: float) -> np.ndarray:
    """Green's function k_f(x,s) — Eq. (6.3)."""
    denom = math.sqrt(4.0 * math.pi * alpha * dt)
    diff = x[:, None] - s[None, :]
    return np.exp(-(diff**2) / (4.0 * alpha * dt)) / denom


def project_kernel_to_basis(
    kernel_mat: np.ndarray,
    u_grid: np.ndarray,
    dx: float,
    d: int = 1,
) -> np.ndarray:
    """L2-projection Λ*_K — Eq. (4.9) stub via Riemann quadrature."""
    m = u_grid.shape[0]
    jk = u_grid @ kernel_mat @ u_grid.T * (dx * dx)
    if d > 1:
        jk = np.kron(np.eye(d), jk)
    lam_u = gram_matrix(u_grid, dx)
    kron_inv = np.kron(np.eye(d), np.linalg.pinv(lam_u))
    return kron_inv @ jk @ kron_inv


def squared_exponential_cov(x: np.ndarray, x_p: np.ndarray, *, amp: float, length: float) -> np.ndarray:
    """Initial / process GP covariance — Sec. 6.1.2."""
    diff = x[:, None] - x_p[None, :]
    return amp * np.exp(-(diff**2) / (2.0 * length * length))


def wave_gaussian_delta(x: np.ndarray, s: np.ndarray, *, c: float, dt: float, eps: float) -> np.ndarray:
    """Smoothed k_φφ for wave equation — Sec. 6.2.1."""
    left = np.exp(-((x[:, None] - (s[None, :] - c * dt)) ** 2) / (2 * eps * eps))
    right = np.exp(-((x[:, None] - (s[None, :] + c * dt)) ** 2) / (2 * eps * eps))
    return 0.5 * (left + right)
