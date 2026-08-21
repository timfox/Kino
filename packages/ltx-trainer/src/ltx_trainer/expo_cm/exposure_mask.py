"""Soft exposure partition (Eq. 5–7)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ExposureMaskConfig:
    tau: float = 0.02
    q_lo: float = 0.02
    q_hi: float = 0.98


def luminance(rgb: Tensor) -> Tensor:
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


def compute_exposure_masks(ldr: Tensor, cfg: ExposureMaskConfig | None = None) -> dict[str, Tensor]:
    """Return ``wover, wunder, wgood`` each ``[B,1,H,W]`` (or ``[1,H,W]`` for 3D input)."""
    cfg = cfg or ExposureMaskConfig()
    squeeze = False
    if ldr.dim() == 3:
        ldr = ldr.unsqueeze(0)
        squeeze = True
    y = luminance(ldr)
    flat = y.reshape(y.shape[0], -1)
    qlo = torch.quantile(flat, cfg.q_lo, dim=1, keepdim=True).view(-1, 1, 1, 1)
    qhi = torch.quantile(flat, cfg.q_hi, dim=1, keepdim=True).view(-1, 1, 1, 1)
    band = (qhi - qlo).clamp(min=1e-6)
    lcore = qlo + cfg.tau * band
    hcore = qhi - cfg.tau * band
    mlow = ((lcore - y) / (cfg.tau * band)).clamp(0.0, 1.0)
    mhigh = ((y - hcore) / (cfg.tau * band)).clamp(0.0, 1.0)
    wover = mhigh * (1.0 - mlow)
    wunder = mlow * (1.0 - mhigh)
    wgood = 1.0 - torch.maximum(wover, wunder)
    out = {"wover": wover, "wunder": wunder, "wgood": wgood, "luminance": y}
    if squeeze:
        out = {k: v.squeeze(0) for k, v in out.items()}
    return out
