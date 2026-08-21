"""Weather Estimation Module — WEM (Sec. III-C)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.cadenet.schema import WeatherCondition


@dataclass
class WEMResult:
    condition: WeatherCondition
    severity: float
    delta_top2: float
    used_clip: bool = False


def _lab_stats(rgb: Tensor) -> tuple[float, float, float, float, float]:
    """Approximate LAB/HSV features on ``(C,H,W)`` in [0,1]."""
    gray = rgb.mean(dim=0)
    mu_l = float(gray.mean()) * 255.0
    sigma_l = float(gray.std()) * 255.0
    mu_s = float(rgb.std(dim=0).mean()) * 255.0
    # edge density proxy via gradients
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    edges = (gx.abs() > 0.05).float().mean() + (gy.abs() > 0.05).float().mean()
    rho_e = float(edges) * 0.5
    rv = float((gy.abs().mean() / (gx.abs().mean() + 1e-6)).clamp(max=10.0))
    return mu_l, sigma_l, mu_s, rho_e, rv


def classify_weather(
    frame: Tensor,
    *,
    clip_label: WeatherCondition | None = None,
    clip_threshold: float = 0.15,
) -> WEMResult:
    """Heuristic WEM with optional CLIP disambiguation."""
    mu_l, sigma_l, mu_s, rho_e, rv = _lab_stats(frame)
    scores = {
        WeatherCondition.FOG: 1.0 if sigma_l < 35 and rho_e < 0.1 else 0.0,
        WeatherCondition.RAIN: 1.0 if rv > 3.0 and mu_s < 60 else 0.0,
        WeatherCondition.SAND: 0.5 if mu_s < 40 and sigma_l < 45 else 0.0,
        WeatherCondition.SNOW: 0.8 if mu_l > 160 and sigma_l < 40 else 0.0,
        WeatherCondition.CLEAR: 0.3,
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ranked[0][1], ranked[1][1]
    delta_top2 = top - second
    condition = ranked[0][0]
    used_clip = False
    if delta_top2 < clip_threshold and clip_label is not None:
        condition = clip_label
        used_clip = True
    if condition == WeatherCondition.FOG:
        severity = max(0.0, min(1.0, 1.0 - sigma_l / 35.0))
    elif condition == WeatherCondition.RAIN:
        severity = max(0.0, min(1.0, rv / 5.0))
    else:
        severity = 0.5
    return WEMResult(condition, severity, delta_top2, used_clip)
