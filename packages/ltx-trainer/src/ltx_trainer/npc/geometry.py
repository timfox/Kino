"""Reachable output polytope geometry (Sec. II-B, Eq. 4–11)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def row_stochastic(W: np.ndarray, *, tol: float = 1e-9) -> bool:
    """Return True if ``W`` is (k, m) row-stochastic."""
    if W.ndim != 2:
        return False
    if np.any(W < -tol):
        return False
    row_sums = W.sum(axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=tol, rtol=0.0) and np.all(W >= 0))


def pmin(W: np.ndarray) -> float:
    """``p_min = min_{x,y} W(y|x)`` (Assumption 1)."""
    return float(np.min(W))


def output_polytope_vertices(W: np.ndarray) -> np.ndarray:
    """Rows of ``W`` as vertices of ``P_W = conv{W(·|x)}`` — shape (k, m)."""
    if not row_stochastic(W):
        raise ValueError("W must be row-stochastic")
    return np.asarray(W, dtype=np.float64)


def affine_output_dimension(W: np.ndarray) -> int:
    """``d = dim P_W = rank(W) - 1`` (Sec. II-B, Eq. 5)."""
    r = int(np.linalg.matrix_rank(W))
    return max(r - 1, 0)


def num_transfer_directions(d: int) -> int:
    """``R_d = d(d+1)`` ordered elementary directions (Eq. 22)."""
    return d * (d + 1)


def stars_and_bars_count(N: int, d: int) -> int:
    """``|L_N| = C(N+d, d)`` for the standard ``d``-simplex lattice (Lemma 1)."""
    if N < 0 or d < 0:
        raise ValueError("N and d must be nonnegative")
    return math.comb(N + d, d)


def log_stars_and_bars(N: int, d: int) -> float:
    """``log |L_N| = d log N + o(1)`` leading term (Lemma 1, Eq. 15)."""
    if N <= 0:
        return float("-inf")
    return d * math.log(N) - math.lgamma(d + 1)


def log_message_count_leading(N: int, d: int, lambda_star: float) -> float:
    """Leading ``log |U_N|`` from Lemma 1 (Eq. 15): ``d log N + log λ* - log d!``."""
    return log_stars_and_bars(N, d) + math.log(max(lambda_star, 1e-300))


def bsc_relative_volume_ratio(delta: float) -> float:
    """``λ*_W = 1 - 2δ`` for BSC reachable interval ``[δ, 1-δ]`` (Sec. VI-A)."""
    if not 0.0 < delta < 0.5:
        raise ValueError("delta must be in (0, 0.5)")
    return 1.0 - 2.0 * delta


def channel_3x4_example() -> np.ndarray:
    """Strictly positive 3×4 channel from Sec. VI-B."""
    return np.array(
        [
            [0.80, 0.05, 0.05, 0.10],
            [0.05, 0.80, 0.05, 0.10],
            [0.05, 0.05, 0.80, 0.10],
        ],
        dtype=np.float64,
    )


def channel_3x4_affine_map(u: Sequence[float]) -> np.ndarray:
    """``T(u) = (0.9 u1, 0.9 u2, 0.9 u3, 0.1)`` for ``u ∈ Δ²`` (Sec. VI-B)."""
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (3,):
        raise ValueError("u must have length 3")
    if u.sum() <= 0:
        raise ValueError("u must be positive sum")
    u = u / u.sum()
    return np.array([0.9 * u[0], 0.9 * u[1], 0.9 * u[2], 0.1], dtype=np.float64)


def channel_3x4_coordinate_map(q: np.ndarray) -> np.ndarray:
    """``ẽH(p) = (p1/0.9, p2/0.9, p3/0.9)`` — Sec. VI-B."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError("q must have length 4")
    return q[:3] / 0.9


def channel_3x4_kw_bounds() -> tuple[float, float]:
    """Per-coordinate lower bound ``1/18`` on ``K_W ⊂ Δ²`` (Sec. VI-B)."""
    return 1.0 / 18.0, 1.0


def channel_3x4_relative_volume_ratio() -> float:
    """``λ*_W = 25/36`` for the Sec. VI-B example."""
    return 25.0 / 36.0


def channel_3x4_transfer_variance(u: np.ndarray, i: int, j: int) -> float:
    """``V_{ij}(u) = (u_i+u_j)/0.9 - (u_i-u_j)²`` for ``i≠j`` (Sec. VI-B)."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if i == j or i not in (0, 1, 2) or j not in (0, 1, 2):
        raise ValueError("i, j must be distinct indices in {0,1,2}")
    return float((u[i] + u[j]) / 0.9 - (u[i] - u[j]) ** 2)


def log_kl_covering_bound(rho: float, d: int, lambda_star: float) -> float:
    """Lemma 3 (Eq. 29): ``log N_D(P_W, ρ) ≤ d log(1/√ρ) + log λ*_W + O_W(1)``."""
    if rho <= 0 or d < 1:
        raise ValueError("rho must be positive and d >= 1")
    return d * math.log(1.0 / math.sqrt(rho)) + math.log(max(lambda_star, 1e-300))
