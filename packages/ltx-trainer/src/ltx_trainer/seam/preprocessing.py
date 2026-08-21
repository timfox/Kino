"""Uniform waveform preprocessing stub (§3.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.seam.config import SeamConfig


def preprocess_waveform(
    wave: np.ndarray,
    *,
    cfg: SeamConfig | None = None,
) -> np.ndarray:
    """DC removal → HPF → loudness norm → peak limit → clamp."""
    cfg = cfg or SeamConfig()
    x = wave.astype(np.float64)
    x = x - np.mean(x)
    # First-order HPF proxy at 70 Hz (stub)
    alpha = 0.97
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    rms = float(np.sqrt(np.mean(y**2)) + 1e-8)
    target = 10 ** (cfg.target_lufs / 20.0) * 0.01
    y = y * (target / rms)
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > cfg.peak_limit:
        y = y * (cfg.peak_limit / peak)
    return np.clip(y, -1.0, 1.0)


def preprocessing_demo(*, seed: int = 0, cfg: SeamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeamConfig()
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 0.05, size=cfg.sample_rate_hz * 2)
    raw += 0.02  # DC offset
    proc = preprocess_waveform(raw, cfg=cfg)
    return {
        "sample_rate_hz": cfg.sample_rate_hz,
        "hpf_hz": cfg.hpf_hz,
        "target_lufs": cfg.target_lufs,
        "peak_limit": cfg.peak_limit,
        "raw_rms": float(np.sqrt(np.mean(raw**2))),
        "proc_peak": float(np.max(np.abs(proc))),
        "proc_mean": float(np.mean(proc)),
    }
