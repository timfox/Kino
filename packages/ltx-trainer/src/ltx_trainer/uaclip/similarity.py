"""CLIP and Utility-Aware CLIP scores (Eq. 2–3)."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def clip_similarity(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
) -> float | np.ndarray:
    """Cosine similarity s_θ(v,t) = u_v(v)^T u_t(t) (Eq. 2)."""
    iv = l2_normalize(image_emb)
    tv = l2_normalize(text_emb)
    if iv.ndim == 1 and tv.ndim == 1:
        return float(np.dot(iv, tv))
    return iv @ tv.T


def utility_aware_score(
    clip_sim: float | np.ndarray,
    visual_utility: float | np.ndarray,
    *,
    alpha_visual: float = 1.0,
    beta_semantic: float = 1.0,
) -> float | np.ndarray:
    """U S_θ,w(v,t) = α_v h_v(v) + β_s s_θ(v,t) (Eq. 3)."""
    return alpha_visual * visual_utility + beta_semantic * clip_sim
