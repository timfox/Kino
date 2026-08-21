"""GRPO advantages and clipped loss (Eq. 2–4)."""

from __future__ import annotations

import math

import numpy as np


def group_advantages(rewards: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Group-relative advantages (Eq. 2)."""
    mu = float(np.mean(rewards))
    sigma = float(np.std(rewards))
    if sigma < eps:
        return np.zeros_like(rewards)
    return (rewards - mu) / (sigma + eps)


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Within-group min–max normalization (Eq. 6)."""
    zmin = float(np.min(values))
    zmax = float(np.max(values))
    if zmax <= zmin:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - zmin) / (zmax - zmin)


def wer_reward(wer: float, *, gamma: float = 1.0) -> float:
    """Intelligibility anchor R_WER = 1 - tanh(γ·WER) (§3.3)."""
    return 1.0 - math.tanh(gamma * wer)


def combined_reward(
    wer: float,
    style_score: float,
    *,
    eta: float = 0.5,
    gamma: float = 1.0,
) -> float:
    """R_k = η R_WER + (1-η) R_style (Eq. 5)."""
    return eta * wer_reward(wer, gamma=gamma) + (1.0 - eta) * style_score


def grpo_token_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    advantages: np.ndarray,
    *,
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> float:
    """Scalar GRPO objective averaged over tokens (Eq. 4 simplified)."""
    ratio = np.exp(log_probs - old_log_probs)
    clipped = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    pg = -np.minimum(ratio * advantages, clipped * advantages)
    kl = beta * (np.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0)
    return float(np.mean(pg + kl))
