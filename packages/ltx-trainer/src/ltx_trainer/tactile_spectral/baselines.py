"""Baseline tactile representations: AR, MFCC, sPeak (Sec. 4.4)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.tactile_spectral.peaks import select_spectral_peaks
from ltx_trainer.tactile_spectral.spectrum import band_mask, magnitude_spectrum, rfft_freqs


def encode_ar(signal: Tensor, *, order: int = 6) -> Tensor:
    """AR coefficients via Yule-Walker proxy (levinson on autocorrelation)."""
    x = signal - signal.mean()
    n = x.numel()
    if n < order + 2:
        return torch.zeros(order, device=signal.device)
    ac = torch.zeros(order + 1, device=signal.device)
    for lag in range(order + 1):
        ac[lag] = (x[: n - lag] * x[lag:]).mean() if lag < n else 0.0
    # Levinson-Durbin
    a = torch.zeros(order, device=signal.device)
    err = ac[0].clone()
    for k in range(order):
        if err.abs() < 1e-12:
            break
        num = ac[k + 1]
        if k > 0:
            num = num - torch.sum(a[:k] * ac[1 : k + 1].flip(0))
        kk = num / err
        a_new = a.clone()
        for j in range(k):
            a_new[j] = a[j] - kk * a[k - 1 - j]
        a_new[k] = kk
        a = a_new
        err = err * (1.0 - kk * kk)
    return a


def synthesize_ar(
    coeffs: Tensor,
    length: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Synthesize texture by filtering white noise with a stabilized AR model."""
    device = device or coeffs.device
    order = coeffs.numel()
    a = coeffs.clone()
    scale = a.abs().sum().clamp(min=1e-8)
    if float(scale) > 0.95:
        a = a * (0.95 / scale)
    noise = torch.randn(length, device=device) * 0.1
    y = torch.zeros(length, device=device)
    for t in range(length):
        acc = noise[t]
        for i in range(min(order, t)):
            acc = acc + a[i] * y[t - i - 1]
        y[t] = acc.clamp(-5.0, 5.0)
    std = y.std().clamp(min=1e-6)
    return 0.2 * y / std


def _mel_filterbank(
    n_fft: int,
    sample_rate: float,
    n_mels: int,
    f_low: float,
    f_high: float,
    *,
    device: torch.device,
) -> Tensor:
    """Simplified log-spaced filterbank for tactile band (MFCC proxy)."""
    freqs = rfft_freqs(n_fft, sample_rate, device=device)
    edges = torch.logspace(
        math.log10(f_low),
        math.log10(f_high),
        n_mels + 2,
        device=device,
    )
    fb = torch.zeros(n_mels, freqs.numel(), device=device)
    for m in range(n_mels):
        f_left, f_center, f_right = edges[m], edges[m + 1], edges[m + 2]
        for k, f in enumerate(freqs):
            fv = float(f.item())
            if f_left <= fv <= f_center:
                fb[m, k] = (fv - float(f_left)) / max(float(f_center - f_left), 1e-6)
            elif f_center < fv <= f_right:
                fb[m, k] = (float(f_right) - fv) / max(float(f_right - f_center), 1e-6)
    return fb


def encode_mfcc(
    signal: Tensor,
    *,
    sample_rate: float,
    n_coeffs: int = 10,
    n_mels: int = 20,
    f_low: float = 20.0,
    f_high: float = 1000.0,
) -> Tensor:
    mag, _ = magnitude_spectrum(signal, sample_rate=sample_rate)
    fb = _mel_filterbank(
        (signal.numel() - 1) * 2,
        sample_rate,
        n_mels,
        f_low,
        f_high,
        device=signal.device,
    )
    if fb.shape[1] != mag.numel():
        fb = torch.nn.functional.interpolate(
            fb.unsqueeze(0),
            size=mag.numel(),
            mode="linear",
            align_corners=False,
        ).squeeze(0)
    power = torch.matmul(fb, mag.pow(2))
    log_power = torch.log(power.clamp(min=1e-10))
    # DCT-II proxy via FFT of log mel energies
    dct = torch.fft.rfft(log_power).real[:n_coeffs]
    return dct


def synthesize_mfcc(
    coeffs: Tensor,
    length: int,
    *,
    sample_rate: float,
    n_mels: int = 20,
    f_low: float = 20.0,
    f_high: float = 1000.0,
) -> Tensor:
    device = coeffs.device
    n_fft = (length - 1) * 2
    fb = _mel_filterbank(n_fft, sample_rate, n_mels, f_low, f_high, device=device)
    log_power = torch.zeros(n_mels, device=device)
    n = min(coeffs.numel(), n_mels)
    log_power[:n] = coeffs[:n].clamp(-10.0, 10.0)
    power = torch.exp(log_power)
    # approximate mel → linear magnitude via filterbank transpose
    mag = torch.matmul(fb.T, power)
    mag = mag / mag.max().clamp(min=1e-8)
    phase = torch.rand(mag.numel(), device=device) * 2 * math.pi
    spec = mag.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
    y = torch.fft.irfft(spec, n=length)
    std = y.std().clamp(min=1e-6)
    return 0.2 * y / std


def encode_speak(
    signal: Tensor,
    *,
    sample_rate: float,
    num_peaks: int = 10,
    f_low: float = 20.0,
    f_high: float = 1000.0,
) -> tuple[Tensor, Tensor]:
    mag, freqs = magnitude_spectrum(signal, sample_rate=sample_rate)
    return select_spectral_peaks(mag, freqs, num_peaks=num_peaks, f_low=f_low, f_high=f_high)


def synthesize_speak(
    peak_freqs: Tensor,
    peak_amps: Tensor,
    length: int,
    *,
    sample_rate: float,
) -> Tensor:
    t = torch.arange(length, device=peak_freqs.device, dtype=peak_freqs.dtype) / sample_rate
    y = torch.zeros(length, device=peak_freqs.device)
    for f, a in zip(peak_freqs, peak_amps):
        y = y + a * torch.cos(2 * math.pi * f * t)
    return y
