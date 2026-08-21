"""Fusion strategies: single-backbone, pool-late, frame-aligned, score average."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ltx_trainer.cpc3_faf.align import adaptive_map_to_length, conv_stride_downsample


def project_features(x: np.ndarray, out_dim: int, seed: int = 0) -> np.ndarray:
    """Toy linear projection W @ x."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((x.shape[1], out_dim)) * 0.05
    return x @ w


def pool_late_fuse(
    canary: np.ndarray,
    wavlm: np.ndarray,
    *,
    hidden: int = 192,
    seed: int = 0,
) -> np.ndarray:
    """Utterance-level [z_c; z_w] after mean pooling (eq. 2 stub)."""
    z_c = project_features(canary.mean(axis=0, keepdims=True), hidden, seed).ravel()
    z_w = project_features(wavlm.mean(axis=0, keepdims=True), hidden, seed + 1).ravel()
    return np.concatenate([z_c, z_w])


def frame_aligned_fuse(
    canary: np.ndarray,
    wavlm: np.ndarray,
    *,
    prep: Literal["avg", "conv"] = "conv",
    stride: int = 4,
    hidden: int = 192,
    shift_steps: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Frame-wise concat on Canary timeline after WavLM preparation (eq. 3 stub)."""
    h_c = project_features(canary, hidden, seed)
    if prep == "avg":
        from ltx_trainer.cpc3_faf.align import masked_average_downsample

        prepared = masked_average_downsample(wavlm, stride)
    else:
        prepared = conv_stride_downsample(wavlm, stride)
    if shift_steps:
        from ltx_trainer.cpc3_faf.align import temporal_shift

        prepared = temporal_shift(prepared, shift_steps)
    h_w = adaptive_map_to_length(
        project_features(prepared, hidden, seed + 2),
        h_c.shape[0],
    )
    fused = np.concatenate([h_c, h_w], axis=-1)
    return fused.mean(axis=0)


def uniform_score_average(y_canary: float, y_wavlm: float, weight: float = 0.5) -> float:
    """Scalar ensemble of independently trained predictors."""
    return weight * y_canary + (1.0 - weight) * y_wavlm


def predict_bounded(logit: float) -> float:
    """Sentence score in [0, 100]: y_hat = 100 * sigmoid(r)."""
    return 100.0 / (1.0 + np.exp(-logit))
