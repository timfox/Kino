"""GrowLoop evaluation metrics — pairwise judges and case-set gates."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig


def kendall_tau(scores: list[float], expected_ranks: list[int] | None = None) -> float:
    """Per-case Kendall τ for monotonic tier ordering."""
    n = len(scores)
    if n < 2:
        return 0.0
    ranks = list(range(1, n + 1)) if expected_ranks is None else expected_ranks
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    actual = [0] * n
    for r, idx in enumerate(order):
        actual[idx] = r + 1
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (actual[i] - actual[j]) * (ranks[i] - ranks[j])
            if a > 0:
                concordant += 1
            elif a < 0:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else 0.0


def cliffs_delta(stronger: list[float], weaker: list[float]) -> float:
    """Cliff's δ for adjacent tier pair."""
    if not stronger or not weaker:
        return 0.0
    gt = sum(1 for x in stronger for y in weaker if x > y)
    lt = sum(1 for x in stronger for y in weaker if x < y)
    return (gt - lt) / (len(stronger) * len(weaker))


def tie_aware_accuracy(preds: list[str], gold: list[str]) -> float:
    """Strict three-way match {A, B, TIE}."""
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, gold, strict=True)) / len(preds)


def pair_accuracy(preds: list[str], gold: list[str]) -> float:
    """Non-tie pairs: winner match; ties contribute 0.5."""
    total = 0.0
    n = 0
    for p, g in zip(preds, gold, strict=True):
        if g == "TIE":
            total += 0.5 if p == "TIE" else 0.0
        else:
            total += 1.0 if p == g else 0.0
        n += 1
    return total / n if n else 0.0


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (no scipy)."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        for rank, idx in enumerate(order):
            r[idx] = float(rank + 1)
        return r

    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (den_x * den_y) if den_x * den_y else 0.0


def cascaded_score(fatal: bool, quality_raw: float, *, fatal_penalty: float = 0.0) -> float:
    """Eq. (1): 0 if safety fatal else quality score on [0, 100] scale."""
    if fatal:
        return fatal_penalty
    return quality_raw


def metrics_smoke(cfg: GrowLoopConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    rng_seed = seed
    del rng_seed
    tier_scores = [69.5, 58.1, 46.5, 22.6]
    tau = kendall_tau(tier_scores)
    delta = cliffs_delta([69.5, 70.0], [58.0, 57.5])
    preds = ["A", "B", "TIE", "A", "B"]
    gold = ["A", "B", "TIE", "B", "B"]
    return {
        "kendall_tau_toy": tau,
        "cliffs_delta_toy": delta,
        "tie_aware_acc_toy": tie_aware_accuracy(preds, gold),
        "pair_acc_toy": pair_accuracy(preds, gold),
        "spearman_toy": spearman_rho([1.0, 2.0, 3.0], [1.1, 2.2, 2.9]),
        "cascaded_fatal_zero": cascaded_score(True, 80.0) == 0.0,
        "cascaded_quality_pass": cascaded_score(False, cfg.best_tier_mean) == cfg.best_tier_mean,
    }
