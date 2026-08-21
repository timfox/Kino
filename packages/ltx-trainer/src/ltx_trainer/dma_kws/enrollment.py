"""Enrollment utilities for DMA-KWS stub (arXiv:2605.22120).

Supports:
- Text-only enrollment (speaker-independent)
- Multi-modal enrollment fusion (speaker-dependent) via a toy MAM
"""

from __future__ import annotations

import numpy as np


def text_enroll(phoneme_ids: list[int], dim: int = 16) -> np.ndarray:
    """Map a phoneme id sequence to a deterministic embedding (toy nn.Embedding)."""
    if not phoneme_ids:
        return np.zeros((dim,), dtype=np.float64)
    # simple hashing to fixed vector
    v = np.zeros((dim,), dtype=np.float64)
    for i, pid in enumerate(phoneme_ids):
        v[(pid + 3 * i) % dim] += 1.0
    v /= max(len(phoneme_ids), 1)
    return v


def audio_enroll(audio_embed: np.ndarray, dim: int = 16) -> np.ndarray:
    """Project an audio embedding to enrollment dim (toy)."""
    a = np.asarray(audio_embed, dtype=np.float64).ravel()
    if a.size == 0:
        return np.zeros((dim,), dtype=np.float64)
    if a.size == dim:
        return a
    # deterministic trunc/pad projection
    out = np.zeros((dim,), dtype=np.float64)
    m = min(dim, a.size)
    out[:m] = a[:m]
    if a.size > dim:
        out += 0.1 * a[:dim]
    return out


def mam_fuse_concat(text_proto: np.ndarray, audio_proto: np.ndarray) -> np.ndarray:
    """Fusion strategy (3) Concat: concatenate then mean-pool."""
    t = np.asarray(text_proto, dtype=np.float64).ravel()
    a = np.asarray(audio_proto, dtype=np.float64).ravel()
    if t.size == 0 and a.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if t.size == 0:
        return a
    if a.size == 0:
        return t
    return 0.5 * (t + a) if t.size == a.size else np.concatenate([t, a])


def mam_fuse_cross_attention(text_proto: np.ndarray, audio_proto: np.ndarray) -> np.ndarray:
    """Fusion strategy (4) Cross-attention: toy attention weight from dot product."""
    t = np.asarray(text_proto, dtype=np.float64).ravel()
    a = np.asarray(audio_proto, dtype=np.float64).ravel()
    if t.size == 0:
        return a
    if a.size == 0:
        return t
    d = min(t.size, a.size)
    score = float(np.dot(t[:d], a[:d]))
    w = 1.0 / (1.0 + np.exp(-score))  # sigmoid weight
    # speaker-aware prototype biased toward enrolled audio when aligned
    out = (1.0 - w) * t[:d] + w * a[:d]
    return out

