"""Generative device style Intervention Network (GIN, Sec. 3.2.1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.btscafe.config import BTSCafeConfig


def _random_group_conv2d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shallow non-trainable grouped conv proxy on (F, T) spectrogram."""
    f, t = x.shape
    kernel = rng.choice([(1, 1), (1, 3), (3, 1)])
    kh, kw = kernel
    pad_f = kh // 2
    pad_t = kw // 2
    padded = np.pad(x, ((pad_f, pad_f), (pad_t, pad_t)), mode="edge")
    out = np.zeros_like(x)
    weight = rng.standard_normal(kernel) * 0.1
    for i in range(f):
        for j in range(t):
            patch = padded[i : i + kh, j : j + kw]
            out[i, j] = float(np.sum(patch * weight) + rng.standard_normal() * 0.01)
    return out


def gain_intervention(x: np.ndarray, rng: np.random.Generator, cfg: BTSCafeConfig) -> np.ndarray:
    g = float(rng.uniform(cfg.gain_min, cfg.gain_max))
    return x * g


def frequency_gate(f_bins: int, rng: np.random.Generator, cfg: BTSCafeConfig) -> np.ndarray:
    """Eq. 3 — frequency-wise random gate clipped to [alpha_min, 1]."""
    alpha_f = rng.uniform(cfg.alpha_min, 1.0, size=f_bins)
    return np.clip(alpha_f, cfg.alpha_min, 1.0)


def frobenius_normalize(x: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Eq. 5 — preserve Frobenius norm of reference."""
    ref_norm = float(np.linalg.norm(ref, ord="fro"))
    x_norm = float(np.linalg.norm(x, ord="fro")) + eps
    return x * (ref_norm / x_norm)


def gin_augment(
    spectrogram: np.ndarray,
    *,
    cfg: BTSCafeConfig | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Eqs. 3–5: content-preserving style perturbation on (F, T) features.

    Returns augmented spectrogram and intermediate diagnostics.
    """
    cfg = cfg or BTSCafeConfig()
    rng = np.random.default_rng(seed)
    x = np.asarray(spectrogram, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("spectrogram must be 2-D (freq, time)")

    x_gain = gain_intervention(x, rng, cfg)
    x_style = _random_group_conv2d(x_gain, rng)
    alpha = frequency_gate(x.shape[0], rng, cfg)[:, None]
    x_blend = alpha * x_gain + (1.0 - alpha) * x_style
    x_aug = frobenius_normalize(x_blend, x_gain)

    return {
        "augmented": x_aug,
        "gain": float(np.mean(x_gain) / (np.mean(x) + 1e-8)),
        "alpha_mean": float(np.mean(alpha)),
        "fro_norm_ratio": float(np.linalg.norm(x_aug, ord="fro") / (np.linalg.norm(x_gain, ord="fro") + 1e-8)),
    }
