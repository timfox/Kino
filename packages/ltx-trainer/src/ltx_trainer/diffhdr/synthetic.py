"""LDR video synthesis from HDR (Sec. 3.1)."""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor

from ltx_trainer.diffhdr.log_gamma import inverse_log_gamma_map, log_gamma_map
from ltx_trainer.diffhdr.mask import detect_exposure_masks
from ltx_trainer.diffhdr.noise import heteroscedastic_noise


def srgb_oetf(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    a = 0.055
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + a) * x ** (1 / 2.4) - a)


def exposure_shift(linear: Tensor, delta_stops: float) -> Tensor:
    return linear * (2.0**delta_stops)


def hdr_to_ldr_video(
    hdr: Tensor,
    *,
    delta_stops: float | None = None,
    apply_noise: bool = True,
    quantize: bool = True,
) -> Tensor:
    """
    Simulate LDR formation: exposure → noise → sRGB → clip → 8-bit.

    hdr: [T,C,H,W] linear radiance
    """
    delta = delta_stops if delta_stops is not None else random.uniform(-2, 2)
    scaled = exposure_shift(hdr, delta)
    if apply_noise:
        scaled = heteroscedastic_noise(scaled)
    srgb = srgb_oetf(scaled.clamp(0.0, 1.0))
    srgb = srgb.clamp(0.0, 1.0)
    if quantize:
        srgb = (srgb * 255).round() / 255.0
    return srgb


def synthesize_hdr_clip(
    frames: int = 8,
    size: int = 64,
    *,
    seed: int = 0,
) -> Tensor:
    """Synthetic HDR video clip [T,C,H,W] with highlight/shadow regions."""
    g = torch.Generator().manual_seed(seed)
    clip = []
    for t in range(frames):
        base = torch.rand(3, size, size, generator=g) * 0.5 + 0.2
        # Moving highlight
        cx = size // 3 + t % 5
        cy = size // 2
        base[0, cy : cy + 6, cx : cx + 6] = 3.0 + 0.5 * torch.sin(torch.tensor(t * 0.3))
        base[1:, size // 4, size // 4] = 0.02  # shadow corner
        clip.append(base.clamp(min=0.0))
    return torch.stack(clip)


def synthesize_pair(
    frames: int = 8,
    size: int = 64,
    *,
    seed: int = 0,
    apply_noise: bool = True,
) -> tuple[Tensor, Tensor]:
    hdr = synthesize_hdr_clip(frames, size, seed=seed)
    ldr = hdr_to_ldr_video(hdr, delta_stops=random.Random(seed).uniform(-1.5, 1.5), apply_noise=apply_noise)
    return ldr, hdr


def vae_roundtrip_error(
    hdr: Tensor,
    *,
    mapping: str = "log_gamma",
    m: float = 16.0,
    gamma: float = 2.2,
) -> Tensor:
    """Map HDR → VAE domain → inverse; used for Log-Gamma ablation (Tab. 4)."""
    if mapping == "linear":
        mapped = hdr.clamp(0.0, 1.0)
        recon = mapped
    elif mapping == "log":
        mapped = torch.log1p(hdr) / math.log1p(m)
        recon = torch.expm1(mapped * math.log1p(m))
    elif mapping == "log_gamma":
        mapped = log_gamma_map(hdr, m=m, gamma=gamma)
        recon = inverse_log_gamma_map(mapped, m=m, gamma=gamma)
    else:
        mapped = torch.log1p(hdr) / torch.log1p(torch.tensor(m))
        recon = inverse_log_gamma_map(log_gamma_map(mapped * m, m=m, gamma=1.0), m=m, gamma=1.0)
    return (recon - hdr).abs().mean()
