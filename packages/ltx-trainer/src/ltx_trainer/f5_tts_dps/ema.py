"""Exponential moving average of model parameters (training stabilizer)."""

from __future__ import annotations

from typing import Mapping


def ema_update(
    shadow: Mapping[str, float],
    online: Mapping[str, float],
    beta: float = 0.99,
) -> dict[str, float]:
    """One EMA step: shadow <- beta * shadow + (1 - beta) * online."""
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    keys = set(shadow) & set(online)
    return {k: beta * shadow[k] + (1.0 - beta) * online[k] for k in keys}


def ema_distance(shadow: Mapping[str, float], online: Mapping[str, float]) -> float:
    """Mean absolute parameter gap (toy convergence diagnostic)."""
    keys = set(shadow) & set(online)
    if not keys:
        return 0.0
    return sum(abs(shadow[k] - online[k]) for k in keys) / len(keys)
