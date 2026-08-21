"""Prediction fusion for decoder/local/global branches (toy)."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def joint_fuse(
    decoder_word: np.ndarray,
    local_word: np.ndarray,
    global_utt: np.ndarray,
    severity: np.ndarray,
) -> np.ndarray:
    """Eq. (8) stub: concatenate branch features at word level."""
    g = np.broadcast_to(global_utt, (decoder_word.shape[0], global_utt.shape[0]))
    s = np.broadcast_to(severity, (decoder_word.shape[0], severity.shape[0]))
    return np.concatenate([decoder_word, local_word, g, s], axis=-1)


def predict_word_correctness(joint_repr: np.ndarray, seed: int = 0) -> np.ndarray:
    """Lightweight LN-MLP style proxy returning probabilities."""
    rng = np.random.default_rng(seed)
    h = (joint_repr - joint_repr.mean(axis=-1, keepdims=True)) / (joint_repr.std(axis=-1, keepdims=True) + 1e-8)
    w = rng.standard_normal((h.shape[1], 1)) * 0.05
    logit = (h @ w).squeeze(-1)
    return sigmoid(logit)
