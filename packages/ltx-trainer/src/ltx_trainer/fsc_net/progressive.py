"""Frequency-progressive learning targets (§II-C)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fsc_net.config import FscNetConfig


def sliding_window_average(residual: np.ndarray, window: int) -> np.ndarray:
    """Average residual R(t,f) over frequency with window Wi and zero-padding."""
    r = np.asarray(residual, dtype=np.float64)
    if window <= 1:
        return r.copy()
    pad = window // 2
    padded = np.pad(r, ((0, 0), (pad, pad)), mode="constant")
    out = np.zeros_like(r)
    for f in range(r.shape[-1]):
        out[:, f] = np.mean(padded[:, f : f + window], axis=-1)
    return out


def progressive_targets(
    x_hr_mag: np.ndarray,
    y_hr_mag: np.ndarray,
    *,
    windows: tuple[int, ...] | None = None,
) -> list[np.ndarray]:
    """|Y_i| = |X_HR| + avg(R, Wi) where R = |Y_HR| - |X_HR| (§II-C)."""
    x_hr = np.asarray(x_hr_mag, dtype=np.float64)
    y_hr = np.asarray(y_hr_mag, dtype=np.float64)
    residual = y_hr - x_hr
    wins = windows or FscNetConfig().progressive_windows
    return [x_hr + sliding_window_average(residual, w) for w in wins]


def progressive_demo(*, seed: int = 0, cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    rng = np.random.default_rng(seed)
    f_bins = 128
    t_frames = 32
    x_hr = rng.uniform(0.1, 0.5, (t_frames, f_bins))
    y_hr = x_hr + rng.uniform(0.0, 0.4, (t_frames, f_bins))
    targets = progressive_targets(x_hr, y_hr, windows=cfg.progressive_windows)
    coarse = targets[0]
    fine = targets[-1]
    return {
        "windows": list(cfg.progressive_windows),
        "num_stages": len(targets),
        "coarse_vs_fine_l1": float(np.mean(np.abs(coarse - fine))),
        "final_equals_hr": bool(np.allclose(fine, y_hr)),
    }
