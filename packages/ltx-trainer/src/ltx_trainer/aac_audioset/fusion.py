"""ConvNeXt + AudioSet keyword fusion stubs (arXiv:2606.05717)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.aac_audioset.config import AacAudiosetConfig


def top_k_audioset_keywords(
    logits: np.ndarray,
    *,
    k: int,
) -> list[int]:
    """Select top-K AudioSet class indices from frozen classifier logits."""
    if logits.ndim != 1:
        raise ValueError("logits must be 1-D over AudioSet classes")
    k = min(k, logits.shape[0])
    return np.argsort(logits)[-k:][::-1].tolist()


def fuse_acoustic_semantic(
    acoustic: np.ndarray,
    keyword_embeddings: np.ndarray,
) -> np.ndarray:
    """§2.2: H_f = [H_a; K] ∈ R^{(T+M)×d}."""
    if acoustic.ndim != 2 or keyword_embeddings.ndim != 2:
        raise ValueError("acoustic and keyword_embeddings must be 2-D")
    if acoustic.shape[1] != keyword_embeddings.shape[1]:
        raise ValueError("embedding dimensions must match")
    return np.concatenate([acoustic, keyword_embeddings], axis=0)


def caption_ce_loss(
    log_probs: np.ndarray,
    targets: np.ndarray,
) -> float:
    """§2.3: L_CE = -Σ log p(y_t | y_<t, H_f)."""
    eps = 1e-9
    p = np.clip(log_probs, eps, 1.0)
    return float(-np.mean(np.sum(targets * np.log(p), axis=-1)))


def decode_step_argmax(logits: np.ndarray) -> int:
    """yt = argmax_w p(w | y_<t, H_f) stub."""
    return int(np.argmax(logits))
