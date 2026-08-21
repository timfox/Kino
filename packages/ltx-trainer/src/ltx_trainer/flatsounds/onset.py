"""Onset / energy-envelope detection for FlatSounds alignment (Sec. 3.1, App. C)."""

from __future__ import annotations

import numpy as np


def energy_envelope(
    audio: np.ndarray,
    *,
    sr: int = 16000,
    frame: int = 512,
    hop: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """RMS envelope from STFT magnitudes."""
    y = np.asarray(audio, dtype=np.float64).ravel()
    if y.size < frame:
        pad = frame - y.size
        y = np.pad(y, (0, pad))
    n_frames = 1 + (len(y) - frame) // hop
    env = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = y[i * hop : i * hop + frame]
        spec = np.abs(np.fft.rfft(seg * np.hanning(frame)))
        env[i] = np.sqrt(np.mean(spec**2) + 1e-12)
    times = (np.arange(n_frames) + 0.5) * hop / sr
    return times, env


def detect_onsets(
    audio: np.ndarray,
    *,
    sr: int = 16000,
    min_spacing_s: float = 0.5,
) -> np.ndarray:
    """Peak picking on envelope with minimum spacing (supplement Fig. 6 mitigation)."""
    times, env = energy_envelope(audio, sr=sr)
    if env.size == 0:
        return np.array([], dtype=np.float64)
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)) + 1e-9)
    thresh = med + 3.0 * mad
    peaks: list[int] = []
    for i in range(1, env.size - 1):
        if env[i] > thresh and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            if peaks and (times[i] - times[peaks[-1]]) < min_spacing_s:
                continue
            peaks.append(i)
    return times[np.array(peaks, dtype=int)] if peaks else np.array([], dtype=np.float64)


def adaptive_tolerance_s(n_hits: int, *, base: float = 0.15) -> float:
    """100–250 ms window depending on hit density (paper Sec. 3.1)."""
    if n_hits <= 1:
        return 0.25
    if n_hits >= 5:
        return 0.10
    return base
