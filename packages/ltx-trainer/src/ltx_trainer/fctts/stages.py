"""Two-stage hierarchical spectrogram generation — §3.2.1."""

from __future__ import annotations

import numpy as np


def timbre_stage_blurry(
    phoneme_emb: np.ndarray,
    z_spk: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stage 1: timbre-conditioned blurry log-mel h."""
    base = phoneme_emb.mean(axis=0) if phoneme_emb.ndim > 1 else phoneme_emb
    timbre = np.asarray(z_spk, dtype=np.float64).ravel()[: base.size]
    blur = base + 0.6 * timbre
    return blur + 0.05 * rng.standard_normal(blur.shape)


def style_stage_refine(
    h: np.ndarray,
    cp: np.ndarray,
    rng: np.random.Generator,
    steps: int = 4,
) -> np.ndarray:
    """Stage 2: style-conditioned refinement (toy CFM steps)."""
    x = np.asarray(h, dtype=np.float64).copy()
    style = np.asarray(cp, dtype=np.float64).ravel()
    if style.size < x.size:
        reps = int(np.ceil(x.size / max(style.size, 1)))
        style = np.tile(style, reps)[: x.size]
    else:
        style = style[: x.size]
    for t_idx in range(steps):
        t = (t_idx + 1) / steps
        delta = 0.15 * style * (1.0 - t)
        x = x + delta + 0.02 * rng.standard_normal(x.size)
    return x


def dual_reference_synthesis(
    phoneme_emb: np.ndarray,
    timbre_ref: np.ndarray,
    style_ref: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Inference with separate timbre and style references."""
    h = timbre_stage_blurry(phoneme_emb, timbre_ref, rng)
    x_hat = style_stage_refine(h, style_ref, rng)
    return {"h_blur": h, "x_hat": x_hat}
