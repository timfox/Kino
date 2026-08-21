"""Camera-mimicking SDR degradations on reference latents (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def tone_map_reinhard(sdr_linear: Tensor, percentile: float = 0.98) -> Tensor:
    """Simple Reinhard tone map to [0, 1] for SDR reference."""
    x = sdr_linear.clamp(min=0.0)
    scale = torch.quantile(x.flatten(), percentile).clamp(min=1e-4)
    mapped = x / scale
    return (mapped / (1.0 + mapped)).clamp(0.0, 1.0)


def contrast_clip(sdr: Tensor, clip_lo: float = 0.02, clip_hi: float = 0.98) -> Tensor:
    return sdr.clamp(clip_lo, clip_hi)


def highlight_shadow_blur(sdr: Tensor, *, sigma: float = 2.0, thresh_lo: float = 0.15, thresh_hi: float = 0.85) -> Tensor:
    """Blur extreme luminance regions to erase recoverable detail."""
    if sdr.dim() == 3:
        sdr = sdr.unsqueeze(0)
    y = 0.2126 * sdr[:, 0:1] + 0.7152 * sdr[:, 1:2] + 0.0722 * sdr[:, 2:3]
    mask = ((y < thresh_lo) | (y > thresh_hi)).float()
    k = int(max(3, sigma * 2 + 1))
    blurred = F.avg_pool2d(F.pad(sdr, [k // 2] * 4, mode="reflect"), k, stride=1)
    out = sdr * (1.0 - mask) + blurred * mask
    return out.squeeze(0) if out.shape[0] == 1 else out


def mp4_compression_sim(sdr: Tensor, *, strength: float = 0.35) -> Tensor:
    """Block DCT-like proxy: downscale-upscale + mild quantization."""
    if sdr.dim() == 3:
        sdr = sdr.unsqueeze(0)
    small = F.interpolate(sdr, scale_factor=0.5, mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=sdr.shape[-2:], mode="bilinear", align_corners=False)
    q = (back * 255.0).round() / 255.0
    out = sdr * (1.0 - strength) + q * strength
    return out.squeeze(0) if out.shape[0] == 1 else out


def joint_exposure_shift(sdr: Tensor, hdr: Tensor, ev: float) -> tuple[Tensor, Tensor]:
    scale = 2.0**ev
    return (sdr * scale).clamp(0.0, 1.0), (hdr * scale).clamp(min=0.0)


def apply_reference_degradations(
    sdr: Tensor,
    *,
    contrast: bool = True,
    blur: bool = True,
    mp4: bool = True,
) -> Tensor:
    """Full camera-mimicking pipeline on SDR reference only."""
    out = sdr
    if contrast:
        out = contrast_clip(out)
    if blur:
        out = highlight_shadow_blur(out)
    if mp4:
        out = mp4_compression_sim(out)
    return out.clamp(0.0, 1.0)
