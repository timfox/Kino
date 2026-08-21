"""Retrieval metrics for FORTE benchmarks (arXiv:2606.05812)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    map_at_10: float
    r_at_1: float
    r_at_5: float
    r_at_10: float
    r_at_50: float | None = None


def recall_at_k(relevant: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    hits = len(relevant & top)
    return hits / len(relevant)


def average_precision(relevant: set[str], ranked_ids: list[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def metrics_from_ranking(relevant: set[str], ranked_ids: list[str]) -> RetrievalMetrics:
    return RetrievalMetrics(
        map_at_10=average_precision(relevant, ranked_ids, 10),
        r_at_1=recall_at_k(relevant, ranked_ids, 1),
        r_at_5=recall_at_k(relevant, ranked_ids, 5),
        r_at_10=recall_at_k(relevant, ranked_ids, 10),
        r_at_50=recall_at_k(relevant, ranked_ids, 50) if len(ranked_ids) >= 50 else None,
    )
