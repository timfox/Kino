"""Tied-attention forward stub — Eq. (1)."""

from __future__ import annotations

from typing import Any

import numpy as np


def row_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def activation(a: np.ndarray, mode: str = "softmax") -> np.ndarray:
    if mode == "identity":
        return a / max(a.shape[-1], 1)
    return row_softmax(a, axis=-1)


def attention_predict(
    x: np.ndarray,
    w_ext: np.ndarray,
    w_lora: np.ndarray | None = None,
    *,
    p_rank: int,
    sigma: str = "softmax",
) -> np.ndarray:
    """Eq. (1): y_hat = sigma(A - E_X A) X with tied K=Q and identity values."""
    t, d = x.shape
    scale = max(d, 1)
    p_inv = 1.0 / max(p_rank, 1)
    a_core = (x @ w_ext) @ w_ext.T @ x.T / scale
    if w_lora is not None:
        lor = (x @ w_lora).reshape(t, 1) @ (x @ w_lora).reshape(1, t)
        a_core = a_core + lor
    a_core = a_core * p_inv
    a_mean = np.mean(a_core, axis=0, keepdims=True)
    attn = activation(a_core - a_mean, sigma)
    return attn @ x


def attention_card() -> dict[str, Any]:
    return {
        "eq": "y_hat = sigma(A - E_X A) X",
        "A": "D^{-1} X (P^{-1/2} W W^T + w w^T) X^T",
        "tied": "keys and queries tied; values identity",
        "lora": "rank-one w fine-tuned; extensive-rank W pre-trained then frozen",
    }
