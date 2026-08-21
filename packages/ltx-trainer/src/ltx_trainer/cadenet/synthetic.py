"""Adverse-weather synthetic frames."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cadenet.cape import enhance
from ltx_trainer.cadenet.schema import Detection, WeatherCondition


def _object_box(cx: float, cy: float, size: float, conf: float = 0.9) -> Detection:
    return Detection(cx - size, cy - size, cx + size, cy + size, conf)


def synthesize_scene(
    *,
    weather: WeatherCondition = WeatherCondition.RAIN,
    size: int = 128,
    seed: int = 0,
) -> tuple[Tensor, Tensor, list[Detection]]:
    """Return ``(degraded, clear_proxy, gt_boxes)``."""
    g = torch.Generator().manual_seed(seed)
    clear = torch.rand(3, size, size, generator=g) * 0.4 + 0.3
    clear[:, 40:80, 40:80] += 0.35
    degraded = enhance(clear, weather, severity=0.6)
    if weather == WeatherCondition.RAIN:
        streak = torch.zeros(1, size, size)
        streak[:, :, ::3] = 0.4
        degraded = (degraded * 0.7 + streak.expand_as(degraded) * 0.3).clamp(0, 1)
    gt = [_object_box(size * 0.5, size * 0.5, size * 0.15)]
    return degraded, clear, gt
