"""Eisbach log-barrier on DiT temporal belief space (§3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.eisbach.config import EisbachConfig


def temporal_energy(y: np.ndarray) -> np.ndarray:
    """Channel-mean squared energy over time: y (B, C, T) -> (B, T)."""
    return np.mean(y**2, axis=1)


def belief_distribution(energy: np.ndarray) -> np.ndarray:
    """Softmax belief over time positions."""
    e = energy - np.max(energy, axis=-1, keepdims=True)
    exp = np.exp(e)
    return exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-8)


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    """B = H(p) / log(T) per sample."""
    t = p.shape[-1]
    h = -np.sum(p * np.log(p + 1e-8), axis=-1)
    return h / max(np.log(t), 1e-8)


def log_barrier_weights(b: np.ndarray) -> np.ndarray:
    """α = -log(1-B), w = 1/(1+α)."""
    b = np.clip(b, 0.0, 1.0 - 1e-6)
    alpha = -np.log(1.0 - b)
    return 1.0 / (1.0 + alpha)


def scale_loss(base_loss: float, weights: np.ndarray, *, lam: float) -> float:
    """L = L_base * ((1-λ) + λ * mean(w))."""
    w_mean = float(np.mean(weights))
    return base_loss * ((1.0 - lam) + lam * w_mean)


def compute_barrier(
    y: np.ndarray,
    *,
    lam: float,
    cfg: EisbachConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    energy = temporal_energy(y)
    p = belief_distribution(energy)
    b = normalized_entropy(p)
    w = log_barrier_weights(b)
    return {
        "belief_entropy": b.tolist(),
        "weights": w.tolist(),
        "mean_weight": float(np.mean(w)),
        "lambda": lam,
    }


def barrier_demo(*, seed: int = 0, cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    rng = np.random.default_rng(seed)
    structured = rng.normal(size=(2, 8, 128))
    structured[:, :, 40:60] *= 4.0
    flat = rng.normal(size=(2, 8, 128)) * 0.3
    s = compute_barrier(structured, lam=cfg.barrier_lambda, cfg=cfg)
    f = compute_barrier(flat, lam=cfg.barrier_lambda, cfg=cfg)
    return {
        "structured_mean_weight": s["mean_weight"],
        "flat_mean_weight": f["mean_weight"],
        "structured_gets_higher_weight": s["mean_weight"] > f["mean_weight"],
        "lambda": cfg.barrier_lambda,
    }
