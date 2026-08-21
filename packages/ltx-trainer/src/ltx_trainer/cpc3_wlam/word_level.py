"""Reference-conditioned word-level utilities (toy)."""

from __future__ import annotations

import numpy as np


def aggregate_word_states(decoder_states: np.ndarray, token_to_word: list[list[int]]) -> np.ndarray:
    """Eq. (3) stub: aggregate teacher-forced decoder states on word spans."""
    out = np.zeros((len(token_to_word), decoder_states.shape[1]), dtype=decoder_states.dtype)
    for i, ids in enumerate(token_to_word):
        if not ids:
            continue
        out[i] = decoder_states[ids].mean(axis=0)
    return out


def sentence_intelligibility_from_word_probs(word_probs: np.ndarray, valid_mask: np.ndarray) -> float:
    """Eq. (1): sentence percentage from masked word correctness probabilities."""
    denom = float(valid_mask.sum())
    if denom <= 0:
        return 0.0
    return 100.0 * float((word_probs * valid_mask).sum()) / denom


def masked_bce_loss(target: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> float:
    """Eq. (2) toy masked BCE."""
    eps = 1e-8
    p = np.clip(pred, eps, 1.0 - eps)
    bce = -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
    denom = float(valid_mask.sum())
    if denom <= 0:
        return 0.0
    return float((bce * valid_mask).sum() / denom)
