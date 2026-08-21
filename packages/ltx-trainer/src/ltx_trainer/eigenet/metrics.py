"""Acoustic metrics EDT, C50, T60 for RIR evaluation (Sec. IV-B)."""

from __future__ import annotations

import math

import numpy as np


def _energy_envelope(rir: np.ndarray) -> np.ndarray:
    x = np.asarray(rir, dtype=np.float64).reshape(-1)
    x = x / (np.max(np.abs(x)) + 1e-12)
    return np.cumsum(x**2)


def edt_error(pred: np.ndarray, target: np.ndarray, *, sr: int = 16000) -> float:
    """Early Decay Time error on initial 5 dB drop (energy-envelope stub)."""
    ep = _energy_envelope(pred)
    et = _energy_envelope(target)
    if ep[-1] <= 0 or et[-1] <= 0:
        return float("nan")

    def edt_s(e: np.ndarray) -> float:
        thr = e[-1] * 10 ** (-0.5)
        idx = int(np.searchsorted(e, thr))
        return idx / float(sr)

    return abs(edt_s(ep) - edt_s(et))


def c50_error(pred: np.ndarray, target: np.ndarray, *, sr: int = 16000, early_ms: float = 50.0) -> float:
    """Clarity C50 error in dB (early 50 ms vs late energy)."""

    def c50(x: np.ndarray) -> float:
        e = np.asarray(x, dtype=np.float64).reshape(-1) ** 2
        n_early = max(1, int(sr * early_ms / 1000.0))
        early = e[:n_early].sum()
        late = e[n_early:].sum() + 1e-12
        return 10.0 * math.log10(early / late)

    return abs(c50(np.asarray(pred)) - c50(np.asarray(target)))


def t60_percent_error(pred: np.ndarray, target: np.ndarray, *, sr: int = 16000) -> float:
    """T60 via T20 × 3 approximation (xRIR-style, Sec. IV-B)."""

    def t20(x: np.ndarray) -> float:
        e = np.cumsum(np.asarray(x, dtype=np.float64).reshape(-1) ** 2)
        if e[-1] <= 0:
            return float("nan")
        start = e[-1]
        drop5 = start * 10 ** (-0.5)
        drop25 = start * 10 ** (-2.5)
        i5 = int(np.searchsorted(e, drop5))
        i25 = int(np.searchsorted(e, drop25))
        if i25 <= i5:
            return 0.1
        return 3.0 * (i25 - i5) / float(sr)

    tp, tt = t20(pred), t20(target)
    if math.isnan(tp) or math.isnan(tt) or tt == 0:
        return float("nan")
    return abs(tp - tt) / abs(tt) * 100.0


def metric_triplet(pred: np.ndarray, target: np.ndarray, *, sr: int = 16000) -> dict[str, float]:
    return {
        "EDT": edt_error(pred, target, sr=sr),
        "C50": c50_error(pred, target, sr=sr),
        "T60_pct": t60_percent_error(pred, target, sr=sr),
    }


def sabine_t60_proxy(volume_m3: float, surface_m2: float, alpha_mean: float = 0.2) -> float:
    """Eq. (11) Sabine approximation stub: T60 ∝ V / (c α S)."""
    return 0.161 * volume_m3 / (alpha_mean * surface_m2 + 1e-6)
