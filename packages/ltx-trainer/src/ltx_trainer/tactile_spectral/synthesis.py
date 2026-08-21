"""Time-domain synthesis from compact spectral representations."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.tactile_spectral.sbeta import sbeta_envelope
from ltx_trainer.tactile_spectral.spectrum import magnitude_spectrum, rfft_freqs
from ltx_trainer.tactile_spectral.sslope import sslope_envelope


def synthesize_from_envelope(
    freqs: Tensor,
    envelope: Tensor,
    length: int,
    *,
    sample_rate: float,
    rng: torch.Generator | None = None,
) -> Tensor:
    """Combine magnitude envelope with random phase (Sec. 4.4 sBeta)."""
    phase = torch.rand(envelope.shape, device=envelope.device, generator=rng) * 2 * torch.pi
    spec = envelope.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))
    return torch.fft.irfft(spec, n=length)


def synthesize_sbeta(
    *,
    alpha: float,
    beta: float,
    scale: float,
    length: int,
    sample_rate: float,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    device: torch.device | None = None,
) -> Tensor:
    device = device or torch.device("cpu")
    freqs = rfft_freqs(length, sample_rate, device=device)
    env = sbeta_envelope(freqs, alpha=alpha, beta=beta, f_low=f_low, f_high=f_high, scale=scale)
    return synthesize_from_envelope(freqs, env, length, sample_rate=sample_rate)


def synthesize_sslope(
    *,
    ra: int,
    rb: int,
    f_peak: float,
    scale: float,
    length: int,
    sample_rate: float,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    device: torch.device | None = None,
) -> Tensor:
    device = device or torch.device("cpu")
    freqs = rfft_freqs(length, sample_rate, device=device)
    env = sslope_envelope(
        freqs, ra=ra, rb=rb, f_peak=f_peak, f_low=f_low, f_high=f_high, scale=scale
    )
    noise = torch.randn(length, device=device)
    n_mag, _ = magnitude_spectrum(noise, sample_rate=sample_rate)
    n_mag = n_mag / n_mag.max().clamp(min=1e-8)
    spec = (env[: n_mag.numel()] * n_mag).to(torch.complex64)
    phase = torch.rand(spec.shape, device=device) * 2 * torch.pi
    spec = spec * torch.exp(1j * phase.to(torch.complex64))
    return torch.fft.irfft(spec, n=length)
