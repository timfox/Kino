"""Geometry-informed modulation block + spectrum losses (Sec. III-D, Eq. 2, 10)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.eigenet.config import EigeNetConfig


def multi_octave_spectrum(rir: np.ndarray, *, n_bands: int = 7) -> np.ndarray:
    """Stub 7-band ISO octave power (63 Hz–4 kHz)."""
    x = np.asarray(rir, dtype=np.float64).reshape(-1)
    # Toy band energies from FFT bins grouped by band index
    spec = np.abs(np.fft.rfft(x)) ** 2
    bands = np.array_split(spec, n_bands)
    return np.array([b.mean() for b in bands], dtype=np.float64)


def geometry_informed_modulate(
    target_geom: np.ndarray,
    target_acoustic_proxy: np.ndarray,
    *,
    cfg: EigeNetConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """DiT-style modulation: T0 conditioned on G0 → A0; regress Ŝ0 — Sec. III-D."""
    cfg = cfg or EigeNetConfig()
    g = np.asarray(target_geom, dtype=np.float64).reshape(-1)
    t = np.asarray(target_acoustic_proxy, dtype=np.float64).reshape(-1)
    scale = 1.0 + 0.1 * np.tanh(g[: min(g.size, t.size)].mean())
    a0 = t * scale
    s_hat = multi_octave_spectrum(a0, n_bands=cfg.n_octave_bands)
    return a0, s_hat


def mse_spectrum_loss(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return float(np.mean((p - t) ** 2))


def spectrum_loss(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    cfg: EigeNetConfig | None = None,
) -> dict[str, float]:
    """Eq. (2): L_spectrum = L_MRSTFT + L_EDC (stubbed as MSE + envelope decay)."""
    cfg = cfg or EigeNetConfig()
    lm = mse_spectrum_loss(pred, target)
    # EDC stub: match log-decay of band energies
    lp = np.log(pred + 1e-8)
    lt = np.log(target + 1e-8)
    ledc = float(np.mean((lp - lt) ** 2))
    total = lm + ledc
    return {"L_MRSTFT": lm, "L_EDC": ledc, "L_spectrum": total}


def total_training_loss(
    waveform_loss: float,
    spectrum_loss_val: float,
    *,
    cfg: EigeNetConfig | None = None,
) -> float:
    cfg = cfg or EigeNetConfig()
    return waveform_loss + cfg.lambda_edc * 0.0 + cfg.lambda_spectrum * spectrum_loss_val
