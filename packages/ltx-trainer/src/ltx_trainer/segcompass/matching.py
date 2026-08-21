"""Bipartite matching for multi-object mask rewards (Appendix B.3)."""

from __future__ import annotations


def pairwise_overlap(a: float, b: float) -> float:
    """Soft overlap proxy in [0, 1]."""
    return min(a, b) / max(a, b, 1e-6)


def hungarian_mean_overlap(pred: list[float], gt: list[float]) -> float:
    """Greedy max-overlap assignment — toy for Hungarian matching score."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    remaining_gt = list(gt)
    total = 0.0
    for p in pred:
        if not remaining_gt:
            break
        overlaps = [pairwise_overlap(p, g) for g in remaining_gt]
        best_idx = max(range(len(overlaps)), key=lambda i: overlaps[i])
        total += overlaps[best_idx]
        remaining_gt.pop(best_idx)
    return total / max(len(pred), len(gt))
