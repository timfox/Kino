"""Hard vs soft token assignment (§3.1–3.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ssl_soft_infer.config import AssignmentMode, SslSoftInferConfig


def squared_distances(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """D_k(x) = ||x - c_k||_2^2 for feature x and centroid matrix."""
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    if x.ndim == 1:
        diff = c - x
        return np.sum(diff * diff, axis=1)
    diff = c[None, :, :] - x[:, None, :]
    return np.sum(diff * diff, axis=2)


def hard_token(x: np.ndarray, centroids: np.ndarray) -> int | np.ndarray:
    """Eq. (1): argmin_k D_k(x)."""
    d = squared_distances(x, centroids)
    if d.ndim == 1:
        return int(np.argmin(d))
    return np.argmin(d, axis=1)


def soft_posterior(x: np.ndarray, centroids: np.ndarray, tau: float) -> np.ndarray:
    """Eq. (3): p(k|x) ∝ exp(-D_k(x)/τ)."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    d = squared_distances(x, centroids)
    if d.ndim == 1:
        logits = -d / tau
        logits -= np.max(logits)
        p = np.exp(logits)
        return p / np.sum(p)
    logits = -d / tau
    logits -= np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits)
    return p / np.sum(p, axis=1, keepdims=True)


def token_embedding(
    x: np.ndarray,
    centroids: np.ndarray,
    embeddings: np.ndarray,
    *,
    mode: AssignmentMode = "hard",
    tau: float = 8.0,
) -> np.ndarray:
    """Eq. (2) hard or Eq. (4) soft weighted sum of embeddings."""
    e = np.asarray(embeddings, dtype=np.float64)
    if mode == "hard":
        idx = hard_token(x, centroids)
        if isinstance(idx, np.ndarray):
            return e[idx]
        return e[idx]
    p = soft_posterior(x, centroids, tau)
    if p.ndim == 1:
        return p @ e
    return p @ e


def tau_for_dataset(name: str, cfg: SslSoftInferConfig | None = None) -> float:
    c = cfg or SslSoftInferConfig()
    key = name.lower().replace(" ", "").replace("-", "")
    mapping = {
        "librispeech": c.tau_librispeech,
        "testclean": c.tau_librispeech,
        "testother": c.tau_librispeech,
        "ted2": c.tau_ted2,
        "tedliumv2": c.tau_ted2,
        "chime4": c.tau_chime4,
        "erj": c.tau_erj,
        "ljspeech": c.tau_synth,
        "timit": c.tau_synth,
    }
    return mapping.get(key, c.tau_librispeech)


def posterior_demo(seed: int = 0, cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    """Toy 2-D centroid demo showing soft vs hard embedding."""
    c = cfg or SslSoftInferConfig()
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(4, 8))
    embeddings = centroids.copy()
    x = centroids[2] + rng.normal(scale=0.15, size=8)

    hard_idx = hard_token(x, centroids)
    z_hard = token_embedding(x, centroids, embeddings, mode="hard")
    z_soft = token_embedding(x, centroids, embeddings, mode="soft", tau=c.tau_librispeech)
    p = soft_posterior(x, centroids, c.tau_librispeech)

    return {
        "hard_token": int(hard_idx),
        "soft_top_token": int(np.argmax(p)),
        "soft_entropy": float(-np.sum(p * np.log(p + 1e-12))),
        "embedding_l2_diff": float(np.linalg.norm(z_soft - z_hard)),
        "tau": c.tau_librispeech,
    }
