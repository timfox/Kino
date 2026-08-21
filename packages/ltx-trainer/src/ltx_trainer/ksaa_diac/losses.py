"""R-Drop + Focal loss — §3.2."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def focal_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    gamma: float = 0.34,
    label_smoothing: float = 0.018,
) -> float:
    """Focal loss with label smoothing for per-position diacritic classes."""
    probs = softmax(logits)
    n_class = probs.shape[-1]
    targets = np.asarray(targets, dtype=np.int64)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(targets.size), targets] = 1.0
    smooth = one_hot * (1.0 - label_smoothing) + label_smoothing / n_class
    pt = (probs * smooth).sum(axis=-1)
    return float(-np.mean((1.0 - pt) ** gamma * np.log(pt + 1e-12)))


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    kl_pq = np.sum(p * np.log(p / q), axis=-1)
    kl_qp = np.sum(q * np.log(q / p), axis=-1)
    return float(np.mean(kl_pq + kl_qp))


def rdrop_loss(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    targets: np.ndarray,
    alpha: float = 2.08,
    gamma: float = 0.34,
    label_smoothing: float = 0.018,
) -> dict[str, float]:
    """Focal + α · symmetric KL between two dropout-masked passes."""
    pa = softmax(logits_a)
    pb = softmax(logits_b)
    focal = 0.5 * (
        focal_loss(logits_a, targets, gamma, label_smoothing)
        + focal_loss(logits_b, targets, gamma, label_smoothing)
    )
    kl = symmetric_kl(pa, pb)
    total = focal + alpha * kl
    return {"L_total": total, "L_focal": focal, "L_rdrop_kl": kl, "alpha": alpha}
