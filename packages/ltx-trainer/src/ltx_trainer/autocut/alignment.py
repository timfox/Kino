"""Multimodal alignment stage (frozen backbone, new embeddings)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ltx_trainer.autocut.config import AutoCutConfig


@dataclass
class AlignmentLoss:
    ntp_loss: float
    token_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {"ntp_loss": self.ntp_loss, "token_count": self.token_count}


def next_token_prediction_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """LNTP = -sum log P(x_t | x_<t) — cross-entropy stub on flattened logits."""
    logits = logits.astype(np.float64)
    targets = targets.astype(np.int64).reshape(-1)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    vocab = logits.shape[-1]
    # softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / (exp.sum(axis=-1, keepdims=True) + eps)
    idx = np.clip(targets, 0, vocab - 1)
    p = prob[np.arange(len(idx)), idx]
    return float(-np.mean(np.log(p + eps)))


def alignment_step(
    token_ids: Sequence[int],
    embedding_table: np.ndarray,
    cfg: AutoCutConfig | None = None,
) -> AlignmentLoss:
    """Single alignment minibatch smoke."""
    c = cfg or AutoCutConfig()
    rng = np.random.default_rng(c.random_seed)
    ids = np.asarray(list(token_ids), dtype=np.int64)
    if ids.size == 0:
        ids = rng.integers(0, 1000, size=32)
    emb = embedding_table[ids % embedding_table.shape[0]]
    logits = emb @ rng.normal(size=(emb.shape[1], 256))
    targets = np.roll(ids, -1)
    loss = next_token_prediction_loss(logits, targets)
    return AlignmentLoss(ntp_loss=loss, token_count=int(ids.size))
