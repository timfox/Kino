"""ILD, IPD, and cross-correlation binaural cues (§2.2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.biear.config import BiearConfig

_EPS = 1e-8


def compute_ild(z_l: np.ndarray, z_r: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10((np.abs(z_l) + _EPS) / (np.abs(z_r) + _EPS))


def compute_ipd(z_l: np.ndarray, z_r: np.ndarray) -> np.ndarray:
    return np.angle(z_l * np.conj(z_r))


def compute_cc(wave_l: np.ndarray, wave_r: np.ndarray, *, sample_rate: int, max_lag_ms: float = 3.0) -> np.ndarray:
    max_lag = int(max_lag_ms / 1000.0 * sample_rate)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.array([np.sum(wave_l * np.roll(wave_r, lag)) for lag in lags], dtype=np.float64)
    cc = cc / (np.linalg.norm(cc) + _EPS)
    if cc.shape[0] != 100:
        idx = np.linspace(0, len(cc) - 1, 100).astype(int)
        cc = cc[idx]
    return cc


def binaural_embeddings(
    z_l: np.ndarray,
    z_r: np.ndarray,
    wave_l: np.ndarray,
    wave_r: np.ndarray,
    *,
    cfg: BiearConfig | None = None,
) -> np.ndarray:
    cfg = cfg or BiearConfig()
    ild = compute_ild(z_l, z_r)
    ipd = compute_ipd(z_l, z_r)
    ild_emb = ild[:100] if ild.size >= 100 else np.pad(ild, (0, 100 - ild.size))
    ipd_emb = ipd[:100] if ipd.size >= 100 else np.pad(ipd, (0, 100 - ipd.size))
    cc_emb = compute_cc(wave_l, wave_r, sample_rate=cfg.sample_rate_hz)
    return np.concatenate([ild_emb, ipd_emb, cc_emb])


def binaural_demo(*, seed: int = 0, cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    rng = np.random.default_rng(seed)
    n = int(cfg.segment_s * cfg.sample_rate_hz)
    wave_l = rng.normal(0, 0.1, n)
    wave_r = np.roll(wave_l, 5) * 0.8
    z_l = rng.normal(size=100) + 1j * rng.normal(size=100)
    z_r = z_l * 0.7 * np.exp(1j * 0.2)
    emb = binaural_embeddings(z_l, z_r, wave_l, wave_r, cfg=cfg)
    return {
        "embedding_dim": int(emb.shape[0]),
        "ild_mean_db": float(np.mean(compute_ild(z_l, z_r))),
        "cc_dim": 100,
    }
