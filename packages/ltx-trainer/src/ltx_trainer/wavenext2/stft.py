"""STFT-spec feature for WaveNeXt-based generator — §3.1."""

from __future__ import annotations

import numpy as np


def stft_spec_features(waveform: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    """Real-valued STFT-spec: full real part + imag excluding DC/Nyquist."""
    x = np.asarray(waveform, dtype=np.float64).ravel()
    spec = np.fft.rfft(x, n=n_fft)
    real = np.real(spec)
    imag = np.imag(spec)
    if len(imag) >= 2:
        imag_inner = imag[1:-1]
    else:
        imag_inner = imag
    return np.concatenate([real, imag_inner])


def predict_noise_toy(
    mel: np.ndarray,
    stft_spec: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Toy noise prediction from mel + STFT-spec (stand-in for ConvNeXt sub-model)."""
    mel = np.asarray(mel, dtype=np.float64).ravel()
    stft_spec = np.asarray(stft_spec, dtype=np.float64).ravel()
    scale = 0.01 * (1.0 + float(np.mean(np.abs(mel))))
    return scale * rng.standard_normal(min(len(stft_spec), 256))
