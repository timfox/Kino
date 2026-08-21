"""Friction signal spectrum utilities (Sec. 4.3)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rfft_freqs(n: int, sample_rate: float, device: torch.device | None = None) -> Tensor:
    """One-sided frequency axis for rfft length n."""
    device = device or torch.device("cpu")
    return torch.linspace(0.0, sample_rate / 2.0, n // 2 + 1, device=device)


def magnitude_spectrum(signal: Tensor, *, sample_rate: float) -> tuple[Tensor, Tensor]:
    """|FFT| of 1D friction signal with matching frequency bins."""
    if signal.dim() != 1:
        signal = signal.reshape(-1)
    spec = torch.fft.rfft(signal)
    mag = spec.abs()
    freqs = rfft_freqs(signal.numel(), sample_rate, device=signal.device)
    return mag, freqs


def band_mask(freqs: Tensor, f_low: float, f_high: float) -> Tensor:
    return (freqs >= f_low) & (freqs <= f_high)


def combine_lateral_axes(fx: Tensor, fy: Tensor) -> Tensor:
    """Combine x/y force components into 1D lateral friction (Sec. 4.3)."""
    return torch.sqrt(fx.pow(2) + fy.pow(2))


def hanning_window(length: int, *, device: torch.device | None = None) -> Tensor:
    device = device or torch.device("cpu")
    return torch.hann_window(length, periodic=False, device=device)


def bandpass_fft(signal: Tensor, *, sample_rate: float, f_low: float, f_high: float) -> Tensor:
    """FFT-domain bandpass between f_low and f_high."""
    spec = torch.fft.rfft(signal)
    freqs = rfft_freqs(signal.numel(), sample_rate, device=signal.device)
    mask = band_mask(freqs, f_low, f_high).to(spec.dtype)
    return torch.fft.irfft(spec * mask, n=signal.numel())


def band_energy(mag: Tensor, freqs: Tensor, f_low: float, f_high: float) -> Tensor:
    """Total spectral energy in [f_low, f_high]."""
    m = band_mask(freqs, f_low, f_high)
    if not m.any():
        return torch.tensor(0.0, device=mag.device)
    return mag[m].pow(2).sum()


def log_spaced_band_edges(
    f_low: float,
    f_high: float,
    n_bands: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Log-spaced band edges for critical-band analysis (Fig. 3C)."""
    device = device or torch.device("cpu")
    return torch.logspace(
        math.log10(f_low),
        math.log10(f_high),
        n_bands + 1,
        device=device,
    )
