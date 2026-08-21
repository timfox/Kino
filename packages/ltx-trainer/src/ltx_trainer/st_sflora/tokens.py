"""Semantic token importance, selection, and merging (Sec. IV-B)."""

from __future__ import annotations

import numpy as np


def cls_to_patch_attention(
    query_cls: np.ndarray,
    keys_patch: np.ndarray,
    *,
    temperature: float | None = None,
) -> np.ndarray:
    """Attention from [CLS] to patch tokens (Eq. 11–12), shape (B, N)."""
    q = np.asarray(query_cls, dtype=np.float64)
    k = np.asarray(keys_patch, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if k.ndim == 2:
        k = k.reshape(1, k.shape[0], k.shape[1])
    scale = float(temperature if temperature is not None else np.sqrt(q.shape[-1]))
    logits = np.einsum("bd,bnd->bn", q, k) / scale
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def batch_token_importance(attention_batch: np.ndarray) -> np.ndarray:
    """Batch-level importance vector ᾱ_m (Eq. 18), shape (N,)."""
    a = np.asarray(attention_batch, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("attention_batch must be (B, N)")
    per_sample = np.sort(a, axis=-1)[:, ::-1]
    return per_sample.sum(axis=0)


def cumulative_semantic_retention(importance_sorted: np.ndarray, k: int) -> float:
    """f_m(K) = sum_{n=1}^K ᾱ_{m,n} (Eq. 19)."""
    imp = np.asarray(importance_sorted, dtype=np.float64)
    k = int(max(0, min(k, imp.size)))
    if k == 0:
        return 0.0
    return float(imp[:k].sum())


def payload_bits(
    k_tokens: int,
    *,
    batch_size: int,
    embed_dim: int,
    bits_per_element: int,
) -> float:
    """S_m = B (K+2) D q0 (Eq. 4) — CLS + top-K + merge token."""
    return float(batch_size * (int(k_tokens) + 2) * embed_dim * bits_per_element)


def topk_patch_indices(attention_row: np.ndarray, k: int) -> np.ndarray:
    k = int(max(0, min(k, attention_row.size)))
    if k == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-np.asarray(attention_row, dtype=np.float64))
    return order[:k]


def merge_discarded_tokens(
    patch_tokens: np.ndarray,
    attention_row: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    """Attention-weighted merge of non-selected patches (Eq. 14)."""
    patches = np.asarray(patch_tokens, dtype=np.float64)
    alpha = np.asarray(attention_row, dtype=np.float64)
    all_idx = np.arange(patches.shape[0])
    mask = np.ones(patches.shape[0], dtype=bool)
    mask[selected_idx] = False
    discard = all_idx[mask]
    if discard.size == 0:
        return patches.mean(axis=0)
    w = alpha[discard]
    w_sum = w.sum()
    if w_sum <= 0:
        return patches[discard].mean(axis=0)
    w = w / w_sum
    return (patches[discard] * w[:, None]).sum(axis=0)


def refine_token_sequence_length(k_selected: int) -> int:
    """Sequence length after selection: K + 2 ([CLS], K patches, merge)."""
    return int(k_selected) + 2
