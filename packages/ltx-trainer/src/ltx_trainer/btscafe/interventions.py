"""Fig. 2 empirical style-removal analysis on CLAP embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.btscafe.config import BTSCafeConfig


def background_mean_subtract(embeddings: np.ndarray, device_ids: np.ndarray) -> np.ndarray:
    """Device-specific mean estimated from non-event intervals (proxy: per-device mean)."""
    out = embeddings.copy()
    for dev in np.unique(device_ids):
        mask = device_ids == dev
        if np.any(mask):
            out[mask] -= np.mean(embeddings[mask], axis=0, keepdims=True)
    return out


def low_rank_whiten(embeddings: np.ndarray, rank: int = 2) -> np.ndarray:
    """Low-rank whitening proxy — suppress dominant device-related components."""
    x = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    keep = min(rank, s.size)
    recon = (u[:, :keep] * s[:keep]) @ u[:, :keep].T
    return recon


def knn_accuracy(features: np.ndarray, labels: np.ndarray, k: int = 50) -> float:
    """k-NN accuracy proxy (paper uses k=50 on CLAP embeddings)."""
    n = features.shape[0]
    if n <= k + 1:
        return 0.0
    correct = 0
    for i in range(n):
        dists = np.sum((features - features[i]) ** 2, axis=1)
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, k)[:k]
        votes = labels[nn_idx]
        pred = np.bincount(votes.astype(int), minlength=int(labels.max()) + 1).argmax()
        correct += int(pred == labels[i])
    return correct / n


def embedding_intervention_analysis(
    embeddings: np.ndarray,
    device_ids: np.ndarray,
    disease_labels: np.ndarray,
    *,
    k: int = 50,
    cfg: BTSCafeConfig | None = None,
) -> dict[str, Any]:
    """Reproduce Fig. 2 k-NN trade-off structure on toy or real embeddings."""
    cfg = cfg or BTSCafeConfig()
    raw = {
        "device_acc": knn_accuracy(embeddings, device_ids, k=k),
        "disease_acc": knn_accuracy(embeddings, disease_labels, k=k),
    }
    mean_emb = background_mean_subtract(embeddings, device_ids)
    mean = {
        "device_acc": knn_accuracy(mean_emb, device_ids, k=k),
        "disease_acc": knn_accuracy(mean_emb, disease_labels, k=k),
    }
    white_emb = low_rank_whiten(embeddings)
    white = {
        "device_acc": knn_accuracy(white_emb, device_ids, k=k),
        "disease_acc": knn_accuracy(white_emb, disease_labels, k=k),
    }
    return {"raw": raw, "mean_subtract": mean, "whiten": white, "paper_anchors": {
        "raw": {"device": cfg.raw_device_acc, "disease": cfg.raw_disease_acc},
        "mean": {"device": cfg.mean_device_acc, "disease": cfg.mean_disease_acc},
        "whiten": {"device": cfg.whiten_device_acc, "disease": cfg.whiten_disease_acc},
    }}
