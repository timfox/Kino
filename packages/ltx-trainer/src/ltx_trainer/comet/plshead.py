"""PLSHead spectral truncation and linear projection decoding proxies."""

from __future__ import annotations

import numpy as np

from ltx_trainer.comet.config import CometConfig
from ltx_trainer.comet.pls import center_embeddings, pls_svd, project_coefficients


def plshead_truncate(
    embedding: np.ndarray,
    *,
    mean: np.ndarray,
    directions: np.ndarray,
    head_size: int,
    uv_align: np.ndarray | None = None,
    weighted: bool = False,
) -> np.ndarray:
    """Keep top-K PLS projection coefficients (PLSHead / PLSHeadW)."""
    centered = np.asarray(embedding, dtype=np.float64) - mean
    coef = directions.T @ centered
    head = coef[:head_size].copy()
    if weighted and uv_align is not None:
        head = head * uv_align[:head_size]
    return head


def linear_projection_decoding(
    audio: np.ndarray,
    text_memory: np.ndarray,
    *,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Eq. (11–12): linear PD without softmax — U (X̂^T X̂) (U^T V) â."""
    _, a_mean = center_embeddings(audio.reshape(1, -1))
    _, t_mean = center_embeddings(text_memory)
    a_c = audio - a_mean
    a_hat = v.T @ a_c
    x_hat = project_coefficients(text_memory - t_mean, u)
    rescale = x_hat.T @ x_hat
    filt = u.T @ v
    mapped_hat = rescale @ filt @ a_hat
    return u @ mapped_hat + t_mean


def head_energy_ratio(
    coef: np.ndarray,
    *,
    head_size: int,
) -> dict[str, float]:
    """Eq. (5) style norm split over head vs tail."""
    c = coef.size
    head = coef[:head_size]
    tail = coef[head_size:]
    norm_head = float(np.linalg.norm(head))
    norm_tail = float(np.linalg.norm(tail)) if tail.size else 0.0
    norm_all = float(np.linalg.norm(coef))
    return {
        "norm_head": norm_head,
        "norm_tail": norm_tail,
        "norm_all": norm_all,
        "head_fraction": norm_head / (norm_all + 1e-12),
    }
