"""Photometric MEVM proxy: calibrated linear brackets from a single SDR video."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_MEVM_EVS: tuple[float, ...] = (-4.0, 0.0, 4.0)


def mevm_photometric_brackets(
    sdr_gamma_cfhw: Tensor,
    *,
    evs: tuple[float, ...] = DEFAULT_MEVM_EVS,
    gamma: float = 2.2,
    temporal_smooth: bool = True,
    smooth_kernel: int = 3,
) -> tuple[Tensor, list[float]]:
    """Build γ-encoded exposure brackets ``[N,C,F,H,W]`` from one SDR clip ``[C,F,H,W]``."""
    if sdr_gamma_cfhw.ndim != 4:
        raise ValueError(f"Expected [C,F,H,W], got {tuple(sdr_gamma_cfhw.shape)}")
    sdr = sdr_gamma_cfhw.detach().float().clamp(0.0, 1.0)
    linear = sdr.pow(gamma)
    if temporal_smooth and sdr.shape[1] > 1 and smooth_kernel > 1:
        k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
        pad = k // 2
        frame_lum = linear.mean(dim=(0, 2, 3))
        padded = F.pad(frame_lum.view(1, 1, -1), (pad, pad), mode="replicate")
        smooth_lum = F.avg_pool1d(padded, kernel_size=k, stride=1).squeeze()
        scale = (smooth_lum / frame_lum.clamp(min=1e-6)).clamp(0.25, 4.0).view(1, -1, 1, 1)
        linear = (linear * scale).clamp(0.0, 1.0)

    ev_list = [float(e) for e in evs]
    outs = [(linear * (2.0**ev)).clamp(0.0, 1.0).pow(1.0 / gamma) for ev in ev_list]
    return torch.stack(outs, dim=0), ev_list


def simulate_vae_roundtrip(
    brackets_ncfhw: Tensor,
    *,
    noise_std: float = 0.01,
    downscale: int = 4,
) -> Tensor:
    """Approximate VAE encode/decode blur on bracket tensors."""
    if brackets_ncfhw.ndim != 5:
        raise ValueError(f"Expected [N,C,F,H,W], got {tuple(brackets_ncfhw.shape)}")
    x = brackets_ncfhw.detach().float()
    n, c, f, h, w = x.shape
    flat = x.reshape(n * c * f, 1, h, w)
    if downscale > 1:
        small = F.avg_pool2d(flat, kernel_size=downscale, stride=downscale)
        flat = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    if noise_std > 0:
        flat = (flat + torch.randn_like(flat) * noise_std).clamp(0.0, 1.0)
    return flat.reshape(n, c, f, h, w)
