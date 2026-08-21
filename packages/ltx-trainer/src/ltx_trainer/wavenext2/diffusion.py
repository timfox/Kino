"""Diff-WaveNeXt 2: noise-level sub-modeling — §3.3."""

from __future__ import annotations

import numpy as np


def forward_diffusion_sample(
    x0: np.ndarray,
    a_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """x_t = sqrt(a_t) x_0 + sqrt(1 - a_t) ε."""
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(a_t) * x0 + np.sqrt(max(1.0 - a_t, 0.0)) * eps


def denoise_step_residual(
    y_current: np.ndarray,
    predicted_noise: np.ndarray,
    step_gain: float = 0.5,
) -> np.ndarray:
    """One residual denoising sub-model step (toy subtractive update)."""
    return y_current - step_gain * predicted_noise


def sequential_submodel_inference(
    mel: np.ndarray,
    initial_noise: np.ndarray,
    noise_schedule: tuple[float, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply K sub-models sequentially (Fig. 3)."""
    from ltx_trainer.wavenext2.stft import predict_noise_toy, stft_spec_features

    y = initial_noise.copy()
    for a_t in noise_schedule:
        spec = stft_spec_features(y)
        n_pred = predict_noise_toy(mel, spec, rng)
        y = denoise_step_residual(y, n_pred[: len(y)] if len(n_pred) >= len(y) else np.pad(n_pred, (0, len(y) - len(n_pred))))
    return y


def mse_denoising_loss(x0: np.ndarray, x_pred: np.ndarray) -> float:
    """Training loss surrogate for diffusion sub-model (MSE to clean waveform)."""
    return float(np.mean((x0 - x_pred) ** 2))
