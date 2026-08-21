"""Pairwise win-rate and correlation helpers (Sec. 4.5 human alignment)."""

from __future__ import annotations

import math


def pairwise_outcome(human_pref: str, benchmark_pref: str) -> float:
    """Map paired preferences to Win=1, Loss=0, Tie=0.5 (paper Sec. 4.5)."""
    mapping = {"win": 1.0, "loss": 0.0, "tie": 0.5}
    h = mapping.get(human_pref.lower(), 0.5)
    b = mapping.get(benchmark_pref.lower(), 0.5)
    # Agreement score for a single comparison direction (both expressed as win rates per model)
    if h == b:
        return h
    return 0.5


def model_win_ratio(wins: int, ties: int, losses: int) -> float:
    """``(W + 0.5 T) / (W + T + L)`` for model-level win rates."""
    w, t, loss = int(wins), int(ties), int(losses)
    denom = w + t + loss
    if denom == 0:
        return float("nan")
    return (w + 0.5 * t) / denom


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation for human-alignment validation smoke tests."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)
