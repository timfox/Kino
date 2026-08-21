"""Markov blanket extraction from a discovered DAG adjacency (§4, Fig. 1)."""

from __future__ import annotations

import numpy as np


def markov_blanket_indices(
    adjacency: np.ndarray,
    target_idx: int,
) -> set[int]:
    """
    MB = parents ∪ children ∪ spouses (parents of children).

    ``adjacency[i, j] != 0`` means edge i → j.
    """
    n = adjacency.shape[0]
    parents = {j for j in range(n) if adjacency[j, target_idx] != 0}
    children = {j for j in range(n) if adjacency[target_idx, j] != 0}
    spouses: set[int] = set()
    for c in children:
        for j in range(n):
            if adjacency[j, c] != 0 and j != target_idx:
                spouses.add(j)
    mb = parents | children | spouses
    mb.discard(target_idx)
    return mb


def structural_hamming_distance(adj_a: np.ndarray, adj_b: np.ndarray) -> int:
    """SHD: edge additions + deletions + reversals between two DAG adjacencies."""
    a = (adj_a != 0).astype(int)
    b = (adj_b != 0).astype(int)
    shd = int(np.sum(a != b))
    # Count reversed edges once
    both = (a & b).astype(bool)
    rev = 0
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[1]):
            if both[i, j] and both[j, i]:
                rev += 1
    return shd - rev


def mb_edit_distance(found: set[int], truth: set[int]) -> int:
    """Symmetric difference size vs ground-truth MB."""
    return len(found.symmetric_difference(truth))
