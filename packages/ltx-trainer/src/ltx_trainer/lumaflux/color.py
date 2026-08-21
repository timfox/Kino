"""PQ BT.2020 color transforms (Sec. 3, 5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lumaflux.config import L_MAX_NITS, M2020_LUMA


def pq_eotf(y: Tensor) -> Tensor:
    """ST 2084 PQ EOTF: encoded → linear luminance in [0, L_max]."""
    y = y.clamp(0.0, 1.0)
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 32.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 128.0, 2392.0 / 128.0
    yp = y ** (1.0 / m2)
    num = torch.clamp(yp - c1, min=0.0)
    den = c2 - c3 * yp
    lin = (num / (den + 1e-8)) ** (1.0 / m1)
    return lin * L_MAX_NITS


def pq_oetf(lin: Tensor) -> Tensor:
    """PQ OETF: linear luminance → encoded."""
    x = (lin / L_MAX_NITS).clamp(min=0.0)
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 32.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 128.0, 2392.0 / 128.0
    xp = (c1 + c2 * x**m1) / (1.0 + c3 * x**m1)
    return xp.clamp(0.0, 1.0) ** m2


def bt2020_luma(rgb: Tensor) -> Tensor:
    """Y = m2020^T x (Eq. 6)."""
    r, g, b = M2020_LUMA
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    return r * rgb[:, 0:1] + g * rgb[:, 1:2] + b * rgb[:, 2:3]


def rgb_to_yuv2020(rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    y = bt2020_luma(rgb)
    u = (rgb[:, 1:2] - y) * 0.5 + 0.5
    v = (rgb[:, 0:1] - y) * 0.5 + 0.5
    return y, u.clamp(0, 1), v.clamp(0, 1)


def yuv_to_rgb2020(y: Tensor, u: Tensor, v: Tensor) -> Tensor:
    cb = u - 0.5
    cr = v - 0.5
    r = y + 2.0 * cr
    g = y - 0.344 * cb - 0.714 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


def saturation(rgb: Tensor) -> Tensor:
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    mx = rgb.max(dim=1, keepdim=True).values
    mn = rgb.min(dim=1, keepdim=True).values
    return (mx - mn) / (mx + 1e-6)
