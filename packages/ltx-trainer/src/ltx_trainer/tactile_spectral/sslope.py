"""Spectral slope (sSlope) bandpass model (Sec. 4.4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.tactile_spectral.spectrum import band_mask, magnitude_spectrum


def _quantize_slope_db_per_decade(slope_db: float, step: float = 20.0) -> float:
    return round(slope_db / step) * step


def estimate_sslope_orders(
    mag: Tensor,
    freqs: Tensor,
    *,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    quant_db: float = 20.0,
) -> tuple[int, int, float]:
    """Estimate high-pass order ra, low-pass order rb, and peak frequency."""
    mask = band_mask(freqs, f_low, f_high)
    if not mask.any():
        return 1, 1, (f_low + f_high) / 2.0
    m = mag[mask]
    f = freqs[mask]
    peak_idx = int(torch.argmax(m).item())
    f_peak = float(f[peak_idx].item())

    def slope_db(f1: float, a1: float, f2: float, a2: float) -> float:
        if a1 <= 1e-12 or a2 <= 1e-12 or f1 <= 0 or f2 <= 0:
            return 20.0
        return 20.0 * math.log10(a2 / a1) / math.log10(f2 / f1)

    a_peak = float(m[peak_idx].item())
    a_lo = float(m[0].item())
    a_hi = float(m[-1].item())
    slope_lo = slope_db(f_low, a_lo, f_peak, a_peak)
    slope_hi = slope_db(f_peak, a_peak, f_high, a_hi)
    ra = max(1, int(_quantize_slope_db_per_decade(abs(slope_lo), quant_db) / quant_db))
    rb = max(1, int(_quantize_slope_db_per_decade(abs(slope_hi), quant_db) / quant_db))
    return ra, rb, f_peak


def sslope_envelope(
    freqs: Tensor,
    *,
    ra: int,
    rb: int,
    f_peak: float,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    scale: float = 1.0,
) -> Tensor:
    """Asymmetric triangular bandpass in frequency domain."""
    env = torch.zeros_like(freqs)
    for i, f in enumerate(freqs):
        fv = float(f.item())
        if fv < f_low or fv > f_high:
            continue
        if fv <= f_peak:
            # high-pass side: rise toward peak
            ratio = (fv - f_low) / max(f_peak - f_low, 1e-6)
            env[i] = ratio ** ra
        else:
            ratio = (f_high - fv) / max(f_high - f_peak, 1e-6)
            env[i] = ratio ** rb
    peak = env.max().clamp(min=1e-8)
    return scale * env / peak


def encode_sslope(
    signal: Tensor,
    *,
    sample_rate: float,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    quant_db: float = 20.0,
) -> dict[str, float | int | Tensor]:
    mag, freqs = magnitude_spectrum(signal, sample_rate=sample_rate)
    ra, rb, f_peak = estimate_sslope_orders(
        mag, freqs, f_low=f_low, f_high=f_high, quant_db=quant_db
    )
    m = band_mask(freqs, f_low, f_high)
    scale = float(mag[m].max().item()) if m.any() else 1.0
    return {"ra": ra, "rb": rb, "f_peak": f_peak, "scale": scale, "freqs": freqs}
