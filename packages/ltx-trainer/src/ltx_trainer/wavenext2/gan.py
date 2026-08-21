"""GAN-WaveNeXt 2: fixed-point iteration without initial noise — §3.2."""

from __future__ import annotations

import numpy as np

from ltx_trainer.wavenext2.stft import predict_noise_toy, stft_spec_features


def fixed_point_iteration(
    mel: np.ndarray,
    num_iterations: int,
    rng: np.random.Generator,
    *,
    init_waveform: np.ndarray | None = None,
) -> np.ndarray:
    """T-step WaveFit-style refinement; optional init (paper omits required noise)."""
    mel_len = len(np.asarray(mel).ravel())
    y = init_waveform if init_waveform is not None else np.zeros(mel_len * 300, dtype=np.float64)
    if len(y) < mel_len:
        y = np.pad(y, (0, mel_len - len(y)))
    for _ in range(num_iterations):
        spec = stft_spec_features(y)
        noise_pred = predict_noise_toy(mel, spec, rng)
        n = noise_pred[: len(y)] if len(noise_pred) >= len(y) else np.pad(noise_pred, (0, len(y) - len(noise_pred)))
        y = y - 0.3 * n
    return y


def model_size_millions(num_iterations: int, base_m: float = 14.98) -> float:
    """Approximate parameter count scales with sub-models (Table 1)."""
    per_iter = (59.94 - 29.97) / 3  # 4 vs 2 iter delta ~10M per step from paper
    return base_m + max(0, num_iterations - 1) * per_iter
