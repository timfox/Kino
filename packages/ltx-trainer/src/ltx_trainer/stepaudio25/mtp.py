"""MTP-5 verifiable multi-token decoding utilities (toy).

Paper Eq. for branch weights (H=5, alpha=0.9):
  w_h = alpha^(h-1) / sum_j alpha^(j-1)
Combined loss at position t:
  L_t = CE(p_t, x_{t+1}) + sum_h w_h CE(p_{t,h}, x_{t+1+h})
"""

from __future__ import annotations

import numpy as np


def mtp_branch_weights(h: int = 5, alpha: float = 0.9) -> np.ndarray:
    """Exponentially decayed MTP branch weights."""
    if h <= 0:
        return np.zeros((0,), dtype=np.float64)
    raw = np.array([alpha ** (i) for i in range(h)], dtype=np.float64)
    return raw / float(raw.sum())


def mtp_combined_loss(
    main_ce: float,
    branch_ces: list[float],
    alpha: float = 0.9,
) -> float:
    """Toy scalar combined MTP objective from per-branch cross-entropies."""
    w = mtp_branch_weights(len(branch_ces), alpha=alpha)
    branch = float(np.dot(w, np.asarray(branch_ces, dtype=np.float64)))
    return float(main_ce + branch)


def verify_mtp_prefix(proposed: list[int], reference: list[int]) -> tuple[int, list[int]]:
    """Accept a verified prefix of parallel MTP proposals against autoregressive reference.

    Returns (accepted_length, accepted_tokens).
    """
    if not proposed or not reference:
        return 0, []
    n = min(len(proposed), len(reference))
    accepted: list[int] = []
    for i in range(n):
        if proposed[i] != reference[i]:
            break
        accepted.append(proposed[i])
    # Always accept at least the first token if it matches (or zero if mismatch at start).
    return len(accepted), accepted
