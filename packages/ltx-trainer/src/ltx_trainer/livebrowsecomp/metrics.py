"""pass@k / avg@k and ranking metrics for search-agent eval."""

from __future__ import annotations

import math
from typing import Sequence


def pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    """Unbiased pass@k estimator (single question, n samples, c correct)."""
    if num_samples <= 0 or k <= 0:
        return 0.0
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    fail = 1.0
    for i in range(k):
        fail *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - fail


def pass_at_k_from_bools(correct_flags: Sequence[bool], k: int) -> float:
    c = sum(1 for x in correct_flags if x)
    return pass_at_k(c, len(correct_flags), k)


def avg_at_k(correct_flags: Sequence[bool], k: int) -> float:
    if not correct_flags:
        return 0.0
    use = correct_flags[:k] if len(correct_flags) >= k else correct_flags
    return sum(1.0 for x in use if x) / len(use)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def ranks(vals: Sequence[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg_rank
            i = j + 1
        return r

    return pearson_r(ranks(xs), ranks(ys))
