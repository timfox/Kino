"""Supervised losses and diversity penalties (Eq. 20–26)."""

from __future__ import annotations

import numpy as np


def cross_entropy_logits(logits: np.ndarray, label: int) -> float:
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    return float(-np.log(p[label] + 1e-12))


def mc_loss_from_scores(scores: np.ndarray, label: int) -> float:
    return cross_entropy_logits(np.asarray(scores, dtype=np.float64), label)


def inter_adapter_overlap_penalty(adapters) -> float:
    """Eq. 24: Frobenius norm of Q_i^T Q_j."""
    qs = [a.output_subspace() for a in adapters]
    k = len(qs)
    if k < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += np.linalg.norm(qs[i].T @ qs[j], ord="fro") ** 2
            pairs += 1
    return 2.0 * total / (k * (k - 1))


def direction_diversity_penalty(deltas: list[np.ndarray]) -> float:
    """Eq. 26: mean squared cosine between adapter updates."""
    vecs = []
    for d in deltas:
        n = np.linalg.norm(d)
        if n > 1e-9:
            vecs.append(d / n)
    if len(vecs) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            c = float(np.dot(vecs[i], vecs[j]))
            total += c * c
            pairs += 1
    return 2.0 * total / (len(vecs) * (len(vecs) - 1))
