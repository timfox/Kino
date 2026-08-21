"""Condition-Adaptive Parameterised Enhancement — CAPE (Sec. III-E)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cadenet.schema import WeatherCondition
from ltx_trainer.cadenet.wem import _lab_stats


def _clahe_l_channel(rgb: Tensor, *, clip: float = 2.0, tile: int = 8) -> Tensor:
    """Simple CLAHE proxy on luminance."""
    l = rgb.mean(dim=0, keepdim=True)
    mean = F.avg_pool2d(l.unsqueeze(0), tile, stride=tile)
    mean_up = F.interpolate(mean, size=l.shape[-2:], mode="bilinear", align_corners=False)
    enhanced = (l - mean_up.squeeze(0)).clamp(-1, 1) * clip + l
    enhanced = enhanced.clamp(0, 1)
    return torch.cat([enhanced.expand_as(rgb[:1]), rgb[1:]], dim=0) if rgb.shape[0] == 3 else enhanced


def dark_channel_prior(rgb: Tensor, *, severity: float = 0.5, omega: int = 15) -> Tensor:
    """Fog DCP inversion stub (Eq. 2)."""
    alpha = 0.5 + 0.4 * severity
    dark = rgb.min(dim=0, keepdim=True).values
    dark = -F.max_pool2d(-dark.unsqueeze(0), omega, stride=1, padding=omega // 2).squeeze(0)
    flat = dark.flatten()
    k = max(1, int(0.001 * flat.numel()))
    atmospheric = flat.topk(k).values.mean()
    transmission = (1.0 - alpha * (dark / (atmospheric + 1e-6))).clamp(0.1, 1.0)
    restored = (rgb - atmospheric * (1.0 - transmission)) / transmission
    out = restored.clamp(0, 1)
    return _clahe_l_channel(out, clip=2.0)


def morphological_derain(rgb: Tensor) -> Tensor:
    """5-stage rain derain stub."""
    gray = rgb.mean(dim=0, keepdim=True)
    med = F.avg_pool2d(gray.unsqueeze(0), 5, stride=1, padding=2)
    diff = (gray.unsqueeze(0) - med).abs()
    streak = (diff > 0.08).float()
    rho_rain = float(streak.mean())
    out = rgb.clone()
    if 0.001 < rho_rain < 0.30:
        out = out * (1.0 - streak.squeeze(0)) + med.squeeze(0).expand_as(rgb) * streak.squeeze(0)
    elif rho_rain >= 0.30:
        out = 0.6 * rgb + 0.4 * med.squeeze(0).expand_as(rgb)
    mu_l, _, _, _, _ = _lab_stats(out)
    if mu_l < 130:
        gamma = max(1.05, min(1.40, 130.0 / max(mu_l, 30.0)))
        out = out.pow(1.0 / gamma)
    out = _clahe_l_channel(out, clip=1.5)
    return out.clamp(0, 1)


def enhance(frame: Tensor, condition: WeatherCondition, *, severity: float = 0.5) -> Tensor:
    """Route to CAPE branch."""
    if condition == WeatherCondition.RAIN:
        return morphological_derain(frame)
    if condition == WeatherCondition.FOG:
        return dark_channel_prior(frame, severity=severity)
    if condition in (WeatherCondition.SAND, WeatherCondition.SNOW):
        return _clahe_l_channel(frame, clip=2.0)
    return frame
