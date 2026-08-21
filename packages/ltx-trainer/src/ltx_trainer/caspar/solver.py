"""Levenberg-Marquardt + block Jacobi PCGNR solver stub (§V)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class SolverStats:
    iterations: int
    final_cost: float
    pcg_iters: int
    converged: bool


def block_jacobi_preconditioner(J: Array, block_size: int) -> Array:
    """Diagonal block Jacobi preconditioner M^{-1} (one block per node)."""
    n = J.shape[1]
    inv_diag = np.ones(n)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        blk = J[:, start:end]
        g = blk.T @ blk + 1e-6 * np.eye(end - start)
        inv_diag[start:end] = 1.0 / (np.diag(g) + 1e-9)
    return inv_diag


def pcgnr(
    J: Array,
    r: Array,
    *,
    precond: Array,
    tol: float = 1e-3,
    max_iter: int = 200,
) -> tuple[Array, int]:
    """
    Preconditioned CG on normal equations J^T J dx = -J^T r (PCGNR stub).
    """
    b = -J.T @ r
    M = precond
    x = np.zeros_like(b)
    z = M * b
    p = z.copy()
    rz_old = float(np.dot(b, z))
    if rz_old == 0:
        return x, 0
    for it in range(1, max_iter + 1):
        Ap = J.T @ (J @ p)
        alpha = rz_old / (float(np.dot(p, Ap)) + 1e-12)
        x = x + alpha * p
        z = M * (b - J.T @ (J @ x))
        rz_new = float(np.dot(b - J.T @ (J @ x), z))
        if np.sqrt(abs(rz_new)) < tol * np.sqrt(abs(float(np.dot(b, b))) + 1e-12):
            return x, it
        p = z + (rz_new / rz_old) * p
        rz_old = rz_new
    return x, max_iter


def levenberg_marquardt_step(
    J: Array,
    r: Array,
    *,
    lam: float,
    precond: Array,
    pcg_tol: float,
    pcg_max_iter: int,
) -> tuple[Array, int]:
    """Solve (J^T J + lam I) dx = -J^T r via PCGNR on damped normal equations."""
    J_d = np.vstack([J, np.sqrt(lam) * np.eye(J.shape[1])])
    r_d = np.concatenate([r, np.zeros(J.shape[1])])
    return pcgnr(J_d, r_d, precond=precond, tol=pcg_tol, max_iter=pcg_max_iter)


def lm_solve(
    J_fn,
    r_fn,
    x0: Array,
    *,
    initial_trust: float = 100.0,
    pcg_tol: float = 1e-3,
    max_iter: int = 20,
    block_size: int = 9,
) -> tuple[Array, SolverStats]:
    """
    Quality-based LM dampening stub (Madsen et al.).

    ``J_fn(x)`` and ``r_fn(x)`` return Jacobian and residuals at ``x``.
    """
    x = x0.copy()
    lam = initial_trust
    pcg_total = 0
    for it in range(max_iter):
        r = r_fn(x)
        J = J_fn(x)
        cost = 0.5 * float(np.dot(r, r))
        pre = block_jacobi_preconditioner(J, block_size)
        dx, pcg_it = levenberg_marquardt_step(
            J, r, lam=lam, precond=pre, pcg_tol=pcg_tol, pcg_max_iter=200
        )
        pcg_total += pcg_it
        x_new = x + dx
        r_new = r_fn(x_new)
        cost_new = 0.5 * float(np.dot(r_new, r_new))
        if cost_new < cost:
            x = x_new
            lam = max(lam * 0.5, 1e-6)
            if np.linalg.norm(dx) < 1e-8:
                return x, SolverStats(it + 1, cost_new, pcg_total, True)
        else:
            lam = lam * 2.0
    return x, SolverStats(max_iter, 0.5 * float(np.dot(r_fn(x), r_fn(x))), pcg_total, False)
