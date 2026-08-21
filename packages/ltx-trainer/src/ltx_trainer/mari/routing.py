"""Competitive adapter routing (Eq. 7–11, 27–28)."""

from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def mc_option_entropy(option_logits: np.ndarray) -> float:
    """Eq. 27: entropy over multiple-choice options."""
    return entropy(softmax(option_logits))


def training_winner(losses: list[float]) -> int:
    """Eq. 7: argmin_k ℓ_k."""
    return int(np.argmin(losses))


def inference_entropy_route(option_logits_per_adapter: list[np.ndarray]) -> int:
    """Eq. 10: pick adapter with lowest option entropy."""
    scores = [mc_option_entropy(z) for z in option_logits_per_adapter]
    return int(np.argmin(scores))


def usage_balance_penalty(usage_fractions: np.ndarray) -> float:
    """Eq. 23: L_bal."""
    k = usage_fractions.size
    target = 1.0 / k
    return float(np.sum((usage_fractions - target) ** 2))
