"""Bottleneck adapter, age router, and FiLM conditioning stubs (arXiv:2606.05440)."""

from __future__ import annotations

import numpy as np


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def bottleneck_adapter(
    hidden: np.ndarray,
    down: np.ndarray,
    up: np.ndarray,
    *,
    activation: str = "relu",
) -> np.ndarray:
    """§3 Eq. (2): residual bottleneck adapter after Conformer block."""
    if hidden.ndim != 2:
        raise ValueError("hidden must be (seq, dim)")
    z = layer_norm(hidden) @ down.T
    if activation == "relu":
        z = np.maximum(z, 0.0)
    delta = z @ up.T
    return hidden + delta


def age_router_logits(pooled: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """§3 Eq. (6): z_age = R_ffn(MeanPool(h_bar_4))."""
    if pooled.ndim != 1:
        raise ValueError("pooled must be 1-D")
    h = np.maximum(pooled @ w1.T + b1, 0.0)
    return h @ w2.T + b2


def route_top_k(
    posteriors: np.ndarray,
    *,
    k: int,
) -> tuple[list[int], np.ndarray]:
    """Return top-k age group indices and normalized router weights."""
    if posteriors.ndim != 1:
        raise ValueError("posteriors must be 1-D over age groups")
    k = min(k, posteriors.shape[0])
    idx = np.argsort(posteriors)[-k:][::-1]
    weights = posteriors[idx]
    weights = weights / np.clip(weights.sum(), 1e-9, None)
    return idx.tolist(), weights


def combine_encoder_representations(
    reps: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """§3 Eq. (5): H_top-k = Σ_a w_a H(a)."""
    if not reps:
        raise ValueError("reps must be non-empty")
    stacked = np.stack(reps, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def film_modulate(
    bottleneck: np.ndarray,
    age_embedding: np.ndarray,
    gamma_w: np.ndarray,
    beta_w: np.ndarray,
    gate_logit: float,
) -> np.ndarray:
    """§3 Eq. (7): FiLM scale/shift with learnable gate σ(g_l)."""
    gamma = age_embedding @ gamma_w
    beta = age_embedding @ beta_w
    gate = 1.0 / (1.0 + np.exp(-gate_logit))
    modulated = bottleneck + gate * (gamma * bottleneck + beta - bottleneck)
    return modulated


def transducer_nll(log_probs: np.ndarray, targets: np.ndarray) -> float:
    """§3: L_ASR = -log p(y | x) stub."""
    eps = 1e-9
    p = np.clip(log_probs, eps, 1.0)
    return float(-np.mean(np.sum(targets * np.log(p), axis=-1)))
