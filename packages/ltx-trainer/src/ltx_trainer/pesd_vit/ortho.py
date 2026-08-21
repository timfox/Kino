"""Orthogonal subspace decoupling loss (Eq. 1)."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def orthogonal_decoupling_loss(
    down_projections: Mapping[str, np.ndarray],
) -> float:
    """
    L_ortho = sum_{i≠j} ||A_i^T A_j||_F^2.

    Each A_t is (d, r) down-projection for task t.
    """
    names = list(down_projections.keys())
    if len(names) < 2:
        return 0.0
    total = 0.0
    for i, ni in enumerate(names):
        ai = down_projections[ni]
        for nj in names[i + 1 :]:
            aj = down_projections[nj]
            gram = ai.T @ aj
            total += float(np.sum(gram**2))
    return total


def subspace_overlap_matrix(
    down_projections: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Pairwise ||A_i^T A_j||_F for diagnostics."""
    names = list(down_projections.keys())
    out: dict[str, dict[str, float]] = {n: {} for n in names}
    for i, ni in enumerate(names):
        for nj in names[i:]:
            gram = down_projections[ni].T @ down_projections[nj]
            val = float(np.linalg.norm(gram, ord="fro"))
            out[ni][nj] = val
            out[nj][ni] = val
    return out
