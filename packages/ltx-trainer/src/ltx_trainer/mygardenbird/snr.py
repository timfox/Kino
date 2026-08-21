"""Percentile-based SNR estimation stub (§4.1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mygardenbird.config import MygardenbirdConfig


def frame_rms(waveform: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = max(len(waveform) // frame_len, 1)
    trimmed = waveform[: n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def estimate_snr_db(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    frame_ms: float = 50.0,
    noise_percentile: float = 10.0,
) -> float:
    """SNR via 10th-percentile noise floor on 50 ms RMS frames."""
    frame_len = max(int(sample_rate * frame_ms / 1000.0), 1)
    rms = frame_rms(waveform, frame_len)
    noise_floor = float(np.percentile(rms, noise_percentile))
    signal_level = float(np.max(rms))
    if noise_floor <= 0:
        return 0.0
    return float(20.0 * np.log10(signal_level / noise_floor))


def snr_demo(*, seed: int = 0, cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MygardenbirdConfig()
    rng = np.random.default_rng(seed)
    sr = cfg.sample_rate_16k
    n = int(sr * cfg.clip_duration_s)
    clean = rng.normal(0, 0.05, size=n)
    t = np.linspace(0, cfg.clip_duration_s, n, endpoint=False)
    clean[int(0.4 * n) : int(0.7 * n)] += 0.4 * np.sin(2 * np.pi * 1200 * t[int(0.4 * n) : int(0.7 * n)])
    noisy = clean + rng.normal(0, 0.02, size=n)
    snr = estimate_snr_db(noisy, sample_rate=sr)
    return {
        "estimated_snr_db": snr,
        "within_paper_range": cfg.snr_min_db <= snr <= cfg.snr_max_db or snr > 5.0,
        "paper_mean_db": cfg.snr_mean_db,
    }
