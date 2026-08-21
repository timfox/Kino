"""Camera-mimicking SDR degradations (Sec. 3.2)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DegradationConfig:
    mp4_proxy: bool = True
    contrast_clip: bool = True
    highlight_shadow_blur: bool = True
    contrast_strength: float = 1.35
    blur_sigma: float = 2.0
    mp4_downscale: int = 4
    jpeg_noise: float = 0.02


def _luma(rgb: Tensor) -> Tensor:
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


def apply_sdr_degradations(sdr: Tensor, cfg: DegradationConfig | None = None) -> Tensor:
    """Corrupt SDR reference to force prior-based HDR synthesis in extremes."""
    cfg = cfg or DegradationConfig()
    x = sdr.clamp(0.0, 1.0)
    if cfg.mp4_proxy:
        x = _mp4_compression_proxy(x, downscale=cfg.mp4_downscale, noise=cfg.jpeg_noise)
    if cfg.contrast_clip:
        x = _contrast_clip(x, strength=cfg.contrast_strength)
    if cfg.highlight_shadow_blur:
        x = _selective_extreme_blur(x, sigma=cfg.blur_sigma)
    return x.clamp(0.0, 1.0)


def apply_exposure_shift_pair(sdr: Tensor, hdr: Tensor, ev: float) -> tuple[Tensor, Tensor]:
    """Joint exposure shift preserving physical correspondence."""
    scale = 2.0 ** ev
    s = (sdr * scale).clamp(0.0, 1.0)
    h = (hdr * scale).clamp(min=0.0)
    return s, h


def _mp4_compression_proxy(x: Tensor, *, downscale: int, noise: float) -> Tensor:
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    h, w = x.shape[-2:]
    small = F.interpolate(x, scale_factor=1.0 / downscale, mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    if noise > 0:
        back = back + torch.randn_like(back) * noise
    return back.squeeze(0) if squeeze else back


def _contrast_clip(x: Tensor, *, strength: float) -> Tensor:
    mean = x.mean(dim=(-3, -2, -1), keepdim=True)
    return ((x - mean) * strength + mean).clamp(0.0, 1.0)


def _selective_extreme_blur(x: Tensor, *, sigma: float) -> Tensor:
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    y = _luma(x)
    hi = torch.sigmoid((y - 0.85) / 0.03)
    lo = torch.sigmoid((0.12 - y) / 0.03)
    mask = (hi + lo).clamp(0.0, 1.0)
    k = max(3, int(sigma * 2) | 1)
    blur = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), kernel_size=k, stride=1)
    out = x * (1.0 - mask) + blur * mask
    return out.squeeze(0) if squeeze else out


def random_exposure_ev(*, span: float = 1.0) -> float:
    return random.uniform(-span, span)
