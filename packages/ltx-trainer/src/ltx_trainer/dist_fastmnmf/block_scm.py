"""Block-diagonal SCM helpers (Eq. 11–12)."""

from __future__ import annotations

import numpy as np


def block_diag_scm(blocks: list[np.ndarray]) -> np.ndarray:
    """Construct R_n = blkdiag(R^(1)_n, …, R^(L)_n)."""
    if not blocks:
        raise ValueError("blocks must be non-empty")
    for b in blocks:
        if b.ndim != 2 or b.shape[0] != b.shape[1]:
            raise ValueError("each block must be square 2-D")
    sizes = [b.shape[0] for b in blocks]
    m = int(sum(sizes))
    out = np.zeros((m, m), dtype=np.complex128)
    offset = 0
    for b, sz in zip(blocks, sizes, strict=True):
        out[offset : offset + sz, offset : offset + sz] = b
        offset += sz
    return out


def subarray_mic_index(l: int, mu: int, mic_counts: list[int]) -> int:
    """Global mic index m = μ + Σ_{λ<l} M^(λ) (§III-B)."""
    if l < 0 or l >= len(mic_counts):
        raise IndexError("subarray index out of range")
    if mu < 0 or mu >= mic_counts[l]:
        raise IndexError("within-subarray mic index out of range")
    return int(sum(mic_counts[:l]) + mu)


def per_subarray_joint_diagonalizable(blocks_per_source: list[list[np.ndarray]]) -> bool:
    """Toy check: each subarray's SCM set is jointly diagonalizable (Theorem 1 direction)."""
    if not blocks_per_source:
        return True
    n_sources = len(blocks_per_source)
    n_sub = len(blocks_per_source[0])
    for l in range(n_sub):
        stack = np.stack([blocks_per_source[n][l] for n in range(n_sources)], axis=0)
        s = np.sum(stack, axis=0)
        if np.linalg.matrix_rank(s) < s.shape[0]:
            return False
    return True
