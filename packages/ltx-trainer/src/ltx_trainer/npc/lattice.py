"""Simplex lattice codebooks and coordinate nearest-neighbor decoder (Sec. III)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def enumerate_simplex_lattice(d: int, N: int) -> np.ndarray:
    """Lattice ``L_N = Δ^d ∩ (1/N) Z^{d+1}``; rows sum to 1 (Lemma 1)."""
    if N < 0 or d < 0:
        raise ValueError("N and d must be nonnegative")
    if N == 0:
        u = np.zeros(d + 1, dtype=np.float64)
        u[0] = 1.0
        return u.reshape(1, -1)

    points: list[np.ndarray] = []

    def rec(remaining: int, prefix: list[int]) -> None:
        if len(prefix) == d:
            a0 = remaining
            coeffs = prefix + [a0]
            points.append(np.array(coeffs, dtype=np.float64) / N)
            return
        for a in range(remaining + 1):
            rec(remaining - a, prefix + [a])

    rec(N, [])
    return np.stack(points, axis=0)


def filter_kw_3x4(lattice: np.ndarray, *, lower: float = 1.0 / 18.0) -> np.ndarray:
    """``K_W`` for Sec. VI-B: ``u_i ≥ 1/18`` on ``Δ²``."""
    mask = np.all(lattice[:, :3] >= lower - 1e-12, axis=1)
    return lattice[mask]


def squared_distance(z: np.ndarray, u: np.ndarray) -> float:
    return float(np.sum((np.asarray(z) - np.asarray(u)) ** 2))


def nearest_neighbor_decode(
    z: np.ndarray,
    message_set: np.ndarray,
) -> int | None:
    """
    Decoder ``g_N`` (Eq. 19): index of unique minimizer, or ``None`` on tie / empty set.
    """
    if message_set.size == 0:
        return None
    z = np.asarray(z, dtype=np.float64)
    best_idx: int | None = None
    best_dist = float("inf")
    tie = False
    for idx in range(message_set.shape[0]):
        d = squared_distance(z, message_set[idx])
        if d < best_dist - 1e-15:
            best_dist = d
            best_idx = idx
            tie = False
        elif abs(d - best_dist) <= 1e-15:
            tie = True
    if tie or best_idx is None:
        return None
    return best_idx


def elementary_transfer_violation(
    z: np.ndarray,
    u: np.ndarray,
    N: int,
    d: int,
) -> list[tuple[int, int]]:
    """Directions with ``⟨z-u, e_i-e_j⟩ ≥ 1/N`` (Lemma 2, Eq. 21)."""
    z = np.asarray(z, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    thresh = 1.0 / N
    out: list[tuple[int, int]] = []
    for i in range(d + 1):
        for j in range(d + 1):
            if i == j:
                continue
            inner = (z[i] - u[i]) - (z[j] - u[j])
            if inner >= thresh - 1e-15:
                out.append((i, j))
    return out


def empirical_coordinate_3x4(counts: np.ndarray) -> np.ndarray:
    """``ẽH( bq )`` from Sec. VI-B for count vector length 4."""
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    n = counts.sum()
    if n <= 0:
        raise ValueError("counts must sum to positive n")
    q = counts / n
    return q[:3] / 0.9


def monte_carlo_decode_error(
    n: int,
    N: int,
    message_set: np.ndarray,
    output_laws: Sequence[np.ndarray],
    *,
    trials: int = 300,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Estimate ``P_e(n,N)`` via multinomial counts (Sec. VI-B).

    ``output_laws[u]`` is the length-4 output distribution for message ``u``.
    """
    rng = rng or np.random.default_rng(0)
    if message_set.shape[0] != len(output_laws):
        raise ValueError("message_set and output_laws length mismatch")
    errors = 0
    total = 0
    m = int(output_laws[0].shape[0])
    for u_idx, qu in enumerate(output_laws):
        u = message_set[u_idx]
        qu = np.asarray(qu, dtype=np.float64)
        for _ in range(trials):
            counts = rng.multinomial(n, qu)
            z = empirical_coordinate_3x4(counts)
            dec = nearest_neighbor_decode(z, message_set)
            if dec != u_idx:
                errors += 1
            total += 1
    return errors / max(total, 1)
