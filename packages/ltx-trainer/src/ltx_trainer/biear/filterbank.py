"""ERB Gabor filterbank with adaptive Q-factor (§2.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.biear.config import BiearConfig


def erb_rate(freq_hz: float) -> float:
    return 21.4 * np.log10(1.0 + 0.00437 * freq_hz)


def erb_centers(n_bands: int, f_min: float = 80.0, f_max: float = 8000.0) -> np.ndarray:
    erb_min, erb_max = erb_rate(f_min), erb_rate(f_max)
    rates = np.linspace(erb_min, erb_max, n_bands)
    return np.array([10 ** ((r / 21.4) - 1) / 0.00437 for r in rates])


def delta_q_profile(cfg: BiearConfig, *, relative: bool) -> np.ndarray:
    centers = erb_centers(cfg.n_subbands)
    rates = np.array([erb_rate(f) for f in centers])
    u = (rates - rates.min()) / max(rates.max() - rates.min(), 1e-8)
    if relative:
        base, m_low, m_high = cfg.rel_delta_q_base, cfg.rel_m_low, cfg.rel_m_high
    else:
        base, m_low, m_high = cfg.abs_delta_q_base, cfg.abs_m_low, cfg.abs_m_high
    return base * (m_low + (m_high - m_low) * u)


def apply_q_control(
    q0: np.ndarray,
    delta: np.ndarray,
    delta_q: np.ndarray,
    *,
    relative: bool,
    cfg: BiearConfig | None = None,
) -> np.ndarray:
    cfg = cfg or BiearConfig()
    if relative:
        q = q0 * (1.0 + delta_q * delta)
    else:
        q = q0 + delta_q * delta
    return np.clip(q, cfg.q_min, cfg.q_max)


def gabor_weights(
    freqs: np.ndarray,
    center_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    return np.exp(-((freqs - center_hz) ** 2) / (2.0 * max(bandwidth_hz, 1e-8) ** 2))


def filterbank_demo(*, seed: int = 0, cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    rng = np.random.default_rng(seed)
    q0 = np.linspace(1.0, 8.0, cfg.n_subbands)
    delta = rng.uniform(-1, 1, size=cfg.n_subbands)
    dq = delta_q_profile(cfg, relative=True)
    q_rel = apply_q_control(q0, delta, dq, relative=True, cfg=cfg)
    return {
        "n_subbands": cfg.n_subbands,
        "q_min": cfg.q_min,
        "q_max": cfg.q_max,
        "q_rel_mean": float(np.mean(q_rel)),
        "relative_control": True,
    }
