"""Opacity and blur schedules (TriSplat Sec. 3.3)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.trisplat.config import ProgressiveSharpeningConfig


def opacity_exponent(step: int, cfg: ProgressiveSharpeningConfig, *, total_steps: int = 150000) -> float:
    warm = min(1.0, step / max(cfg.opacity_schedule_steps, 1))
    return cfg.opacity_e_init + warm * (cfg.opacity_e_final - cfg.opacity_e_init)


def map_density_to_opacity(density: np.ndarray, step: int, cfg: ProgressiveSharpeningConfig) -> np.ndarray:
    """Eq. (7): nonlinear density → opacity."""
    p = np.clip(density.astype(np.float64), 0.0, 1.0)
    e = opacity_exponent(step, cfg)
    o = 0.5 * (1.0 - np.power(1.0 - p, e) + np.power(p, e))
    return np.clip(o, cfg.alpha_floor, 1.0).astype(np.float32)


def opacity_temperature_scale(opacity: np.ndarray, step: int, cfg: ProgressiveSharpeningConfig) -> np.ndarray:
    warm = min(1.0, step / max(cfg.opacity_schedule_steps, 1))
    tau = cfg.opacity_tau_init + warm * (cfg.opacity_tau_final - cfg.opacity_tau_init)
    o = np.clip(opacity.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(o / (1.0 - o))
    return (1.0 / (1.0 + np.exp(-tau * logit))).astype(np.float32)


def blur_multiplier(step: int, cfg: ProgressiveSharpeningConfig) -> float:
    warm = min(1.0, step / max(cfg.blur_schedule_steps, 1))
    return cfg.blur_beta_init + warm * (cfg.blur_beta_final - cfg.blur_beta_init)
