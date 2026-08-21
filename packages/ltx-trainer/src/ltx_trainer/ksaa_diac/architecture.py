"""CATT-Whisper prefix fusion — §3.1."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ksaa_diac.config import KsaaDiacConfig


def mean_pool_whisper_frames(
    frames: np.ndarray,
    num_tokens: int = 150,
) -> np.ndarray:
    """Pool 1500 Whisper frames → 150 speech prefix tokens."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 1:
        frames = frames.reshape(-1, 1)
    n_frames, dim = frames.shape
    chunk = max(n_frames // num_tokens, 1)
    pooled = []
    for i in range(num_tokens):
        start = i * chunk
        end = min((i + 1) * chunk, n_frames)
        pooled.append(frames[start:end].mean(axis=0))
    return np.stack(pooled)


def prefix_add_fusion(
    text_tokens: np.ndarray,
    speech_prefix: np.ndarray,
) -> np.ndarray:
    """Add projected speech prefix to dedicated prefix positions before text."""
    text = np.asarray(text_tokens, dtype=np.float64)
    speech = np.asarray(speech_prefix, dtype=np.float64)
    dim = text.shape[-1]
    if speech.shape[-1] != dim:
        speech = speech[:, :dim] if speech.ndim > 1 else speech[:dim]
    n_prefix = min(speech.shape[0], text.shape[0])
    fused = text.copy()
    fused[:n_prefix] = fused[:n_prefix] + speech[:n_prefix]
    return fused


def architecture_summary(cfg: KsaaDiacConfig | None = None) -> dict[str, Any]:
    cfg = cfg or KsaaDiacConfig()
    return {
        "text_encoder": f"CATT Transformer ({cfg.catt_layers} layers, d={cfg.hidden_dim})",
        "speech_encoder": "Whisper-base (frozen in primary config)",
        "fusion": f"prefix addition: {cfg.whisper_frames_pooled} frames → {cfg.whisper_prefix_tokens} tokens",
        "output": f"{cfg.num_diacritic_classes} diacritic classes per Arabic letter",
        "params_m": cfg.total_params_m,
        "trainable_params_m": cfg.trainable_params_m,
    }
