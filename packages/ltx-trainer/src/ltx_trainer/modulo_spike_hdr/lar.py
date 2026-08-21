"""Least absolute remainder (LAR) and modulo operators (Eq. 7–9, Sec. IV-B)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def modulo_image(linear: Tensor, period: float) -> Tensor:
    """Im = mod(I, m) with m = 2^N (Eq. 1)."""
    return torch.remainder(linear, period)


def lar_remainder(o: Tensor, period: float) -> Tensor:
    """R(o) = mod(o + m/2, m) - m/2 (Eq. 8)."""
    half = period * 0.5
    return torch.remainder(o + half, period) - half


def spatial_gradient(x: Tensor) -> tuple[Tensor, Tensor]:
    """Central differences on last two dims; returns (gx, gy)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    gx = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:] - F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]
    gy = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :] - F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]
    return gx, gy


def gradient_magnitude(x: Tensor) -> Tensor:
    gx, gy = spatial_gradient(x)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def laplacian(x: Tensor) -> Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    k = x.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    if x.shape[1] > 1:
        outs = [F.conv2d(x[:, c : c + 1], k, padding=1) for c in range(x.shape[1])]
        return torch.cat(outs, dim=1)
    return F.conv2d(x, k, padding=1)


def poisson_solve_fft(rhs: Tensor) -> Tensor:
    """Approximate Poisson solve Δu = rhs via 2D FFT (stub for P(R(ΔIm)))."""
    squeeze = rhs.dim() == 3
    if squeeze:
        rhs = rhs.unsqueeze(0)
    b, c, h, w = rhs.shape
    out_ch = []
    for ch in range(c):
        f = rhs[:, ch]
        fy = torch.fft.rfftn(f)
        yy = torch.fft.fftfreq(h, device=f.device, dtype=f.dtype)[:, None]
        xx = torch.fft.rfftfreq(w, device=f.device, dtype=f.dtype)[None, :]
        denom = (2 * torch.cos(2 * torch.pi * yy) - 2) + (2 * torch.cos(2 * torch.pi * xx) - 2)
        denom[0, 0] = 1.0
        u = torch.fft.irfftn(fy / denom, s=(h, w))
        u = u - u.mean(dim=(-2, -1), keepdim=True)
        out_ch.append(u.unsqueeze(1))
    out = torch.cat(out_ch, dim=1)
    return out.squeeze(0) if squeeze else out


def lar_gradient_features(modulo: Tensor, period: float) -> Tensor:
    """R(∇Im) per channel (Eq. 7)."""
    squeeze = modulo.dim() == 3
    if squeeze:
        modulo = modulo.unsqueeze(0)
    gx, gy = spatial_gradient(modulo)
    rg = lar_remainder(gx, period)
    rb = lar_remainder(gy, period)
    out = torch.cat([rg, rb], dim=1)
    return out.squeeze(0) if squeeze else out


def lar_laplacian_poisson(modulo: Tensor, period: float) -> Tensor:
    """P(R(ΔIm)) (Eq. 9)."""
    lap = laplacian(modulo)
    rlap = lar_remainder(lap, period)
    return poisson_solve_fft(rlap)
