"""Spike-stream modulo encoding (Sec. V, Eq. 19–24)."""

from __future__ import annotations

import torch
from torch import Tensor


def synthesize_spike_frames(
    irradiance: Tensor,
    *,
    num_frames: int,
    threshold: float = 0.08,
    gain: float = 1.0,
) -> Tensor:
    """Binary spike frames [R, C, H, W] from integrated irradiance (Eq. 19 stub)."""
    if irradiance.dim() == 3:
        irradiance = irradiance.unsqueeze(0)
    c, h, w = irradiance.shape[-3:]
    spikes = torch.zeros(num_frames, c, h, w, device=irradiance.device, dtype=irradiance.dtype)
    residual = irradiance.squeeze(0).clone() * gain
    for r in range(num_frames):
        fire = (residual >= threshold).to(irradiance.dtype)
        spikes[r] = fire
        residual = (residual - threshold * fire).clamp(min=0.0)
        residual = residual + irradiance.squeeze(0) * (threshold * 0.15)
    return spikes


def encode_modulo_from_spikes(
    spikes: Tensor,
    *,
    window: int = 25,
    stride: int = 20,
    period: float = 256.0,
    amplification: float = 15.0,
) -> list[Tensor]:
    """Per-window modulo images I_m^(j,c) (Eq. 24)."""
    r_total = spikes.shape[0]
    out: list[Tensor] = []
    j = 0
    start = 0
    while start + window <= r_total:
        chunk = spikes[start : start + window].sum(dim=0) * amplification
        im = torch.remainder(chunk, period) / period
        out.append(im.clamp(0.0, 1.0))
        j += 1
        start = j * stride
    if not out:
        chunk = spikes.sum(dim=0) * amplification
        out.append(torch.remainder(chunk, period).clamp(0.0, period - 1e-6) / period)
    return out


def exposure_decoupled_fps(
    *,
    readout_hz: float = 20_000.0,
    stride_frames: int = 20,
) -> float:
    """Effective modulo output rate f/P (Sec. V-C)."""
    return readout_hz / max(stride_frames, 1)
