"""Semantic Guided Adaptive Patching (Sec. 4.3.1, Algorithm 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass


def overlap_penalty(boxes: list[tuple[float, float, float, float]]) -> float:
    """Lo(Bk): penalize spatial overlap among cluster bounding boxes (stub IoU sum)."""
    if len(boxes) < 2:
        return 0.0
    total = 0.0
    for i in range(len(boxes)):
        x1, y1, w1, h1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            x2, y2, w2, h2 = boxes[j]
            ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            if union > 0:
                total += inter / union
    return total


def silhouette_score(features: list[list[float]], labels: list[int]) -> float:
    """Ls(Ha, lk) stub: mean cluster cohesion minus separation."""
    if not features or len(set(labels)) < 2:
        return 0.0

    def _dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    n = len(features)
    scores: list[float] = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        other_labels = {labels[j] for j in range(n) if labels[j] != labels[i]}
        if not same or not other_labels:
            continue
        a_in = sum(_dist(features[i], features[j]) for j in same) / len(same)
        b_out = min(
            sum(_dist(features[i], features[j]) for j in range(n) if labels[j] == ol) / max(1, sum(1 for j in range(n) if labels[j] == ol))
            for ol in other_labels
        )
        denom = max(a_in, b_out)
        scores.append((b_out - a_in) / denom if denom > 1e-9 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def clustering_cost(
    k: int,
    boxes: list[tuple[float, float, float, float]],
    features: list[list[float]],
    labels: list[int],
) -> float:
    """L(k) = Lo(Bk) − Ls(Ha, lk) (Eq. 3)."""
    return overlap_penalty(boxes) - silhouette_score(features, labels)


def select_optimal_k(
    *,
    k_min: int,
    k_max: int,
    candidate_costs: dict[int, float] | None = None,
) -> int:
    """k* = argmin L(k) over [k_min, k_max]."""
    if candidate_costs:
        return min(range(k_min, k_max + 1), key=lambda k: candidate_costs.get(k, float("inf")))
    # Default monotonic stub when no features supplied.
    return k_min


def visual_complexity(atomic_features: list[list[float]]) -> float:
    r"""cv(Id,t) = max(0, 1 − (1/|R|) Σ cosim(hi, h̄)) (Eq. 4)."""
    if not atomic_features:
        return 0.0
    dim = len(atomic_features[0])
    centroid = [sum(f[d] for f in atomic_features) / len(atomic_features) for d in range(dim)]

    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0

    sim = sum(_cos(h, centroid) for h in atomic_features) / len(atomic_features)
    return max(0.0, 1.0 - sim)


@dataclass
class TreeNode:
    """Node in adaptive image tree T."""

    node_id: str
    depth: int
    visual_complexity: float
    bbox: tuple[float, float, float, float]
    pruned: bool = False

    def should_keep(self, tau_v: float) -> bool:
        return self.visual_complexity >= tau_v
