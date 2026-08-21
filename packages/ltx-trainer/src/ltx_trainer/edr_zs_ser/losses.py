"""Supervised contrastive + speaker adversarial stubs (arXiv:2606.06200)."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def language_aware_weight(anchor_lang: str, pair_lang: str, *, lam: float) -> float:
    """§2.2 Eq. (7): lambda if cross-lingual same-emotion pair."""
    return lam if pair_lang != anchor_lang else 1.0


def supervised_contrastive_loss(
    embeddings: np.ndarray,
    emotion_labels: np.ndarray,
    language_labels: np.ndarray,
    *,
    temperature: float,
    lam: float,
) -> float:
    """§2.2 Eq. (8): language-aware SupCLR."""
    n = embeddings.shape[0]
    total = 0.0
    for i in range(n):
        pos_idx = [p for p in range(n) if p != i and emotion_labels[p] == emotion_labels[i]]
        if not pos_idx:
            continue
        weights = np.array(
            [language_aware_weight(str(language_labels[i]), str(language_labels[p]), lam=lam) for p in pos_idx]
        )
        pos_logits = np.array([cosine_similarity(embeddings[i], embeddings[p]) / temperature for p in pos_idx])
        all_logits = np.array([cosine_similarity(embeddings[i], embeddings[a]) / temperature for a in range(n) if a != i])
        pos_term = np.sum(weights * np.log(np.exp(pos_logits) / np.sum(np.exp(all_logits)) + 1e-9))
        total += -pos_term / np.sum(weights)
    return float(total / max(n, 1))


def speaker_adversarial_loss(logits: np.ndarray, speaker_ids: np.ndarray) -> float:
    """§2.3 Eq. (10): CE on speaker classifier (feature extractor maximizes via GRL)."""
    eps = 1e-9
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True) + eps
    return float(-np.mean(np.log(probs[np.arange(len(speaker_ids)), speaker_ids.astype(int)] + eps)))


def emotion_ce_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """§2.4 Eq. (11): emotion classification CE stub."""
    eps = 1e-9
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True) + eps
    return float(-np.mean(np.log(probs[np.arange(len(targets)), targets.astype(int)] + eps)))


def total_loss(
    ce: float,
    supclr: float,
    spk_adv: float,
    *,
    alpha: float,
    beta: float,
) -> float:
    """§2.5 Eq. (12): L = LCE + alpha*LSupCLR + beta*LSpkAdv."""
    return ce + alpha * supclr + beta * spk_adv
