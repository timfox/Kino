"""RMS-based segment extraction toy (§3.1.4)."""

from __future__ import annotations

import numpy as np


def window_rms(y: np.ndarray) -> float:
    """RMS amplitude for a window of samples."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))


def rank_windows_by_rms(
    y: np.ndarray,
    *,
    window_samples: int,
    hop_samples: int,
    rms_threshold: float,
) -> list[tuple[int, float]]:
    """Sliding-window RMS ranks; returns (start_sample, rms) for windows above threshold."""
    if window_samples <= 0 or y.size < window_samples:
        return []
    out: list[tuple[int, float]] = []
    for start in range(0, y.size - window_samples + 1, max(1, hop_samples)):
        chunk = y[start : start + window_samples]
        rms = window_rms(chunk)
        if rms >= rms_threshold:
            out.append((start, rms))
    out.sort(key=lambda t: t[1], reverse=True)
    return out
