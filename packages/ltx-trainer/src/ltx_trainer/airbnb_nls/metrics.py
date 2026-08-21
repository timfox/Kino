"""Distribution and evaluation metrics for synthetic search queries."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def word_count_distribution(queries: Sequence[str]) -> dict[int, float]:
    """Normalized histogram of query word counts."""
    counts = Counter(len(q.split()) for q in queries if q.strip())
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in sorted(counts.items())}


def kl_divergence(p: dict[int, float], q: dict[int, float], *, eps: float = 1e-9) -> float:
    """KL(P || Q) over discrete bins (Eq. 1 diagnostic)."""
    keys = set(p) | set(q)
    div = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, eps)
        if pk > 0:
            div += pk * math.log(pk / qk)
    return div


def length_stats(queries: Sequence[str]) -> dict[str, float]:
    """Mean, median, std of word counts (Table 2)."""
    lens = [len(q.split()) for q in queries if q.strip()]
    if not lens:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    lens_sorted = sorted(lens)
    n = len(lens_sorted)
    median = (
        lens_sorted[n // 2]
        if n % 2
        else (lens_sorted[n // 2 - 1] + lens_sorted[n // 2]) / 2
    )
    mean = sum(lens) / n
    var = sum((x - mean) ** 2 for x in lens) / n
    return {"mean": mean, "median": median, "std": math.sqrt(var)}


def pct_in_word_range(queries: Sequence[str], lo: int, hi: int) -> float:
    """Fraction of queries with word count in [lo, hi]."""
    if not queries:
        return 0.0
    ok = sum(1 for q in queries if lo <= len(q.split()) <= hi)
    return ok / len(queries) * 100.0


def pairwise_accuracy(correct: int, total: int) -> float:
    """Pairwise ranking accuracy (Sec. 4.3); 0.5 = random."""
    if total <= 0:
        return 0.5
    return correct / total
