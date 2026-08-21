"""Spherical geodesic slerp for latent resampling (Eq. 7–8)."""

from __future__ import annotations

import torch
from torch import Tensor


def _slerp(z0: Tensor, z1: Tensor, t: Tensor, delta: Tensor) -> Tensor:
    sin_d = torch.sin(delta).clamp_min(1e-6)
    w0 = torch.sin((1 - t) * delta) / sin_d
    w1 = torch.sin(t * delta) / sin_d
    while w0.ndim < z0.ndim:
        w0 = w0.unsqueeze(-1)
        w1 = w1.unsqueeze(-1)
    return w0 * z0 + w1 * z1


def geodesic_latent_interp(z: Tensor, scale: float) -> Tensor:
    """Resample latent grid with longitude then latitude slerp (stub)."""
    if scale == 1.0:
        return z
    b, c, h, w = z.shape
    hr, wr = max(1, int(h * scale)), max(1, int(w * scale))
    z_n = torch.nn.functional.interpolate(z, size=(hr, wr), mode="bilinear", align_corners=False)
    # lightweight geodesic blend along rows
    delta = torch.tensor(0.25, device=z.device, dtype=z.dtype)
    z_up = torch.nn.functional.interpolate(z, size=(h, wr), mode="bilinear", align_corners=False)
    z_lo = z_up.roll(shifts=1, dims=-1)
    t = torch.linspace(0, 1, wr, device=z.device, dtype=z.dtype).view(1, 1, 1, wr)
    z_row = _slerp(z_lo, z_up, t, delta)
    return torch.nn.functional.interpolate(z_row, size=(hr, wr), mode="bilinear", align_corners=False)
