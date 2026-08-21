"""Spectral beta (sBeta) envelope model (Sec. 4.4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.tactile_spectral.peaks import select_spectral_peaks
from ltx_trainer.tactile_spectral.spectrum import band_mask, magnitude_spectrum


def _beta_pdf(x: Tensor, alpha: float, beta: float) -> Tensor:
    """Beta distribution on (0,1), unnormalized-safe for fitting."""
    x = x.clamp(1e-6, 1.0 - 1e-6)
    log_b = (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
    )
    return torch.exp((alpha - 1.0) * torch.log(x) + (beta - 1.0) * torch.log(1.0 - x) + log_b)


def fit_sbeta_params(
    mag: Tensor,
    freqs: Tensor,
    *,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    num_peaks: int = 10,
) -> tuple[float, float]:
    """Estimate α, β from peak anchors on normalized log-frequency axis."""
    peak_f, peak_a = select_spectral_peaks(
        mag, freqs, num_peaks=num_peaks, f_low=f_low, f_high=f_high
    )
    if peak_f.numel() < 2:
        return 2.0, 2.0
    log_f = torch.log10(peak_f.clamp(min=f_low))
    log_lo = math.log10(f_low)
    log_hi = math.log10(f_high)
    x = ((log_f - log_lo) / (log_hi - log_lo)).clamp(1e-4, 1.0 - 1e-4)
    y = peak_a / peak_a.max().clamp(min=1e-8)
    # method-of-moments on weighted samples
    mean = float((x * y).sum() / y.sum().clamp(min=1e-8))
    var = float((y * (x - mean).pow(2)).sum() / y.sum().clamp(min=1e-8))
    var = max(var, 1e-4)
    if var >= mean * (1.0 - mean):
        return 2.0, 2.0
    common = mean * (1.0 - mean) / var - 1.0
    alpha = max(mean * common, 0.5)
    beta = max((1.0 - mean) * common, 0.5)
    return float(alpha), float(beta)


def sbeta_envelope(
    freqs: Tensor,
    *,
    alpha: float,
    beta: float,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    scale: float = 1.0,
) -> Tensor:
    """Beta-shaped magnitude envelope on linear frequency axis."""
    log_f = torch.log10(freqs.clamp(min=1e-3))
    log_lo = math.log10(f_low)
    log_hi = math.log10(f_high)
    x = ((log_f - log_lo) / (log_hi - log_lo)).clamp(1e-6, 1.0 - 1e-6)
    env = _beta_pdf(x, alpha, beta)
    rect = band_mask(freqs, f_low, f_high)
    if bool(rect.any().item()):
        peak = env[rect].max()
    else:
        peak = torch.tensor(1.0, device=freqs.device, dtype=env.dtype)
    return scale * env / peak.clamp(min=1e-8) * rect.to(env.dtype)


def encode_sbeta(
    signal: Tensor,
    *,
    sample_rate: float,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    num_peaks: int = 10,
) -> dict[str, float | Tensor]:
    """Encode friction signal as sBeta parameters + scale."""
    mag, freqs = magnitude_spectrum(signal, sample_rate=sample_rate)
    alpha, beta = fit_sbeta_params(mag, freqs, f_low=f_low, f_high=f_high, num_peaks=num_peaks)
    m = band_mask(freqs, f_low, f_high)
    scale = float(mag[m].max().item()) if m.any() else 1.0
    return {"alpha": alpha, "beta": beta, "scale": scale, "freqs": freqs}
