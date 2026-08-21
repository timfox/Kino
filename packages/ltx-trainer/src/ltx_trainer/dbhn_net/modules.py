"""Interaction, ITB, and TF-CAF CPU stubs."""

from __future__ import annotations

import numpy as np


def interaction_block(ann_in: np.ndarray, snn_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cross-branch exchange (Eq. 38–40 simplified)."""
    pooled = np.mean(snn_in, axis=0, keepdims=True)
    fused = ann_in + snn_in + pooled
    gate = 1.0 / (1.0 + np.exp(-fused))
    out = gate * fused
    return out, out


def information_transformation_block(x: np.ndarray) -> np.ndarray:
    """ITB: spike-to-continuous refinement (Eq. 31–34 simplified)."""
    q1 = np.maximum(0.0, x)
    q2 = np.mean(x, axis=(-2, -1), keepdims=True)
    return q1 * x + q2 * (1.0 - q1)


def tf_cross_attention_fusion(ann: np.ndarray, snn: np.ndarray) -> np.ndarray:
    """TF-CAF dual-domain fusion (Eq. 35–37 simplified)."""
    t_attn = ann + snn * np.tanh(np.mean(snn, axis=-1, keepdims=True))
    f_attn = t_attn + snn * np.tanh(np.mean(t_attn, axis=-2, keepdims=True))
    return 0.5 * (f_attn + ann)


def band_split_spectrum(spec: np.ndarray, n_bands: int = 4) -> list[np.ndarray]:
    """Split complex spectrum along frequency (Eq. 2–4 stub)."""
    f = spec.shape[-1]
    edges = np.linspace(0, f, n_bands + 1, dtype=int)
    return [spec[..., edges[i] : edges[i + 1]] for i in range(n_bands)]


def band_merge_bands(bands: list[np.ndarray]) -> np.ndarray:
    """Reconstruct spectrum from subbands (Eq. 21–22 stub)."""
    return np.concatenate(bands, axis=-1)
