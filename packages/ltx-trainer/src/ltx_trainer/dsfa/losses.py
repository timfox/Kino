"""Joint CE + SupCon loss stub (§3.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dsfa.config import DsfaConfig


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    n = logits.shape[0]
    return float(-np.mean(np.log(probs[np.arange(n), labels.astype(int)] + 1e-8)))


def supcon_loss(embeddings: np.ndarray, labels: np.ndarray, *, temperature: float = 0.07) -> float:
    """Simplified supervised contrastive loss."""
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim = emb @ emb.T / temperature
    n = emb.shape[0]
    loss = 0.0
    count = 0
    for i in range(n):
        pos = (labels == labels[i]) & (np.arange(n) != i)
        if not np.any(pos):
            continue
        logits = sim[i] - np.max(sim[i])
        exp = np.exp(logits)
        denom = np.sum(exp) - exp[i]
        num = np.sum(exp[pos])
        loss += -np.log(num / (denom + 1e-8))
        count += 1
    return float(loss / max(count, 1))


def total_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    *,
    use_supcon: bool,
    cfg: DsfaConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or DsfaConfig()
    ce = cross_entropy_loss(logits, labels)
    sc = supcon_loss(embeddings, labels) if use_supcon else 0.0
    return {
        "ce": ce,
        "supcon": sc,
        "total": ce + cfg.supcon_lambda * sc if use_supcon else ce,
    }


def loss_demo(*, seed: int = 0, cfg: DsfaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DsfaConfig()
    rng = np.random.default_rng(seed)
    n = 8
    logits = rng.normal(size=(n, 2))
    labels = rng.integers(0, 2, size=n)
    emb = rng.normal(size=(n, 64))
    joint = total_loss(logits, labels, emb, use_supcon=True, cfg=cfg)
    ce_only = total_loss(logits, labels, emb, use_supcon=False, cfg=cfg)
    return {
        "lambda_supcon": cfg.supcon_lambda,
        "joint_total": joint["total"],
        "ce_only_total": ce_only["total"],
    }
