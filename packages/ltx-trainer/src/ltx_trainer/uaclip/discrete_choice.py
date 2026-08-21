"""Discrete-choice interpretation of InfoNCE (Section 3.2)."""

from __future__ import annotations

import numpy as np


def conditional_logit_probabilities(utilities: np.ndarray) -> np.ndarray:
    """
    Pr(v_i | t, {v_k}) = exp(V_ii) / sum_k exp(V_ik) with V_ik = utility[i,k].

    utilities: (n_text, n_image) systematic utilities (already scaled by 1/τ if desired).
    """
    u = np.asarray(utilities, dtype=np.float64)
    z = u - u.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def infonce_from_utilities(utilities: np.ndarray) -> float:
    """Negative mean log prob of diagonal matches — equivalent to Eq. 1 with V=s/τ."""
    p = conditional_logit_probabilities(utilities)
    n = p.shape[0]
    return float(-np.mean(np.log(p[np.arange(n), np.arange(n)] + 1e-12)))
