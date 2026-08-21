"""Symbolic vs numerical sparse Cholesky reuse (Sec. 3.2.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CholeskyCache:
    """Stores permutation P and sparsity pattern S_L across NMPC re-solves."""

    permutation: np.ndarray | None = None
    sparsity_pairs: tuple[tuple[int, int], ...] | None = None
    elimination_tree_depth: int = 0
    symbolic_done: bool = False
    numeric_factorizations: int = 0
    symbolic_ms: float = 0.0
    numeric_ms: float = 0.0

    def sparsity_nnz(self) -> int:
        return len(self.sparsity_pairs or ())


def sym_cholesky(k: np.ndarray, *, cost_unit_ms: float) -> CholeskyCache:
    """Eq. (18): compute P, S_L once from sparsity of K_c."""
    n = k.shape[0]
    perm = np.arange(n)
    lower = np.tril(k)
    pairs = tuple((i, j) for i in range(n) for j in range(i + 1))
    depth = int(np.ceil(np.log2(max(n, 2))))
    return CholeskyCache(
        permutation=perm,
        sparsity_pairs=pairs,
        elimination_tree_depth=depth,
        symbolic_done=True,
        symbolic_ms=cost_unit_ms,
    )


def num_cholesky(k: np.ndarray, cache: CholeskyCache, *, cost_unit_ms: float) -> np.ndarray:
    """Eq. (20–21): numeric factorization reusing P and S_L."""
    if not cache.symbolic_done or cache.permutation is None:
        raise RuntimeError("symbolic factorization required before numeric Cholesky")
    p = cache.permutation
    kp = k[np.ix_(p, p)]
    l = np.linalg.cholesky(kp)
    cache.numeric_factorizations += 1
    cache.numeric_ms += cost_unit_ms
    return l


def solve_with_cache(k: np.ndarray, rhs: np.ndarray, cache: CholeskyCache, *, cost_unit_ms: float) -> np.ndarray:
    l = num_cholesky(k, cache, cost_unit_ms=cost_unit_ms)
    y = np.linalg.solve(l, rhs[cache.permutation])
    sol = np.linalg.solve(l.T, y)
    out = np.empty_like(sol)
    out[cache.permutation] = sol
    return out
