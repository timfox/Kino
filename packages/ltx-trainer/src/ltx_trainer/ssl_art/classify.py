"""Classification strategies: zero-shot, KNN zero-shot, linear (Sec. 3.3)."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def zero_shot_predict(
    image_feat: np.ndarray,
    text_feats: np.ndarray,
    labels: list[str],
) -> str:
    """Eq. (1): argmax_k cos(f_I, t_k)."""
    scores = [cosine_similarity(image_feat, text_feats[i]) for i in range(len(labels))]
    return labels[int(np.argmax(scores))]


def knn_zero_shot_predict(
    query_feat: np.ndarray,
    ref_feats: np.ndarray,
    ref_labels: list[str],
    *,
    k: int = 1,
) -> str:
    """Eq. (2): majority vote among top-k reference neighbors."""
    sims = np.array([cosine_similarity(query_feat, ref_feats[i]) for i in range(len(ref_labels))])
    k = min(k, len(ref_labels))
    top_idx = np.argsort(sims)[-k:][::-1]
    votes: dict[str, int] = {}
    for idx in top_idx:
        lab = ref_labels[int(idx)]
        votes[lab] = votes.get(lab, 0) + 1
    return max(votes, key=votes.get)


def linear_predict(
    image_feat: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    labels: list[str],
) -> str:
    """Linear layer on frozen CLS features: argmax z."""
    z = weight @ image_feat.ravel() + bias
    return labels[int(np.argmax(z))]


def accuracy(preds: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for p, g in zip(preds, gold, strict=True) if p == g)
    return correct / len(gold)
