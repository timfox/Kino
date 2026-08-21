"""Retinex-inspired continuous pseudo-pair construction (Sec. 3.1, Eq. 1–2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def srgb_to_linear(rgb: Tensor) -> Tensor:
    """sRGB → linear RGB (per-channel), clamped."""
    x = rgb.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(lin: Tensor) -> Tensor:
    """Linear RGB → sRGB."""
    x = lin.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def luminance_y(rgb: Tensor) -> Tensor:
    """Y = 0.2126 R + 0.7152 G + 0.0722 B (linear or sRGB channels)."""
    if rgb.shape[-3] == 3:
        r, g, b = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    else:
        raise ValueError("expected (..., 3, H, W)")
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def smooth_illumination(y: Tensor, kernel_size: int = 9, sigma: float = 0.15) -> Tensor:
    """Edge-preserving smooth on luminance (bilateral proxy via separable Gaussian)."""
    if y.dim() == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.dim() == 3:
        y = y.unsqueeze(1)
    k = kernel_size | 1
    pad = k // 2
    coords = torch.arange(k, device=y.device, dtype=y.dtype) - pad
    g = torch.exp(-(coords**2) / (2 * (sigma * k) ** 2 + 1e-8))
    g = g / g.sum()
    g_h = g.view(1, 1, 1, -1)
    g_v = g.view(1, 1, -1, 1)
    yh = F.conv2d(y, g_h, padding=(0, pad))
    yv = F.conv2d(yh, g_v, padding=(pad, 0))
    out = yv.squeeze(1)
    if out.dim() == 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out.clamp(min=1e-4)


def decompose_retinex(
    image: Tensor,
    *,
    kernel_size: int = 9,
    sigma: float = 0.15,
) -> tuple[Tensor, Tensor]:
    """Return (reflectance R, illumination L) with I ≈ R ⊙ L in linear space."""
    lin = srgb_to_linear(image.clamp(0, 1))
    y = luminance_y(lin)
    l_map = smooth_illumination(y, kernel_size=kernel_size, sigma=sigma)
    if l_map.dim() == 2:
        l_map = l_map.unsqueeze(0)
    r = (lin / l_map.clamp(min=1e-4)).clamp(0.0, 10.0)
    return r, l_map


def retinex_interpolate(
    i0: Tensor,
    i1: Tensor,
    strength: float,
    *,
    beta_scale: float = 0.5,
) -> Tensor:
    """Eq. (1)–(2): pseudo target I_s from pair (I0, I1)."""
    s = float(strength)
    r0, l0 = decompose_retinex(i0)
    r1, l1 = decompose_retinex(i1)
    beta_s = beta_scale * s
    log_l0 = torch.log(l0.clamp(min=1e-4))
    log_l1 = torch.log(l1.clamp(min=1e-4))
    ls = torch.exp((1.0 - s) * log_l0 + s * log_l1)
    rs = (1.0 - beta_s) * r0 + beta_s * r1
    out_lin = (rs * ls).clamp(0.0, 1.0)
    return linear_to_srgb(out_lin)


def alpha_blend_interpolate(i0: Tensor, i1: Tensor, strength: float) -> Tensor:
    """RGB alpha baseline I_s = (1-s) I0 + s I1 (Sec. 3.1)."""
    s = float(strength)
    return ((1.0 - s) * i0 + s * i1).clamp(0.0, 1.0)


def build_light100k_group(
    i0: Tensor,
    i1: Tensor,
    *,
    strengths: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    use_retinex: bool = True,
    beta_scale: float = 0.5,
) -> dict[float, Tensor]:
    """Construct G = {I_s} including endpoints 0 and 1."""
    group: dict[float, Tensor] = {0.0: i0.clone(), 1.0: i1.clone()}
    interp = retinex_interpolate if use_retinex else alpha_blend_interpolate
    for s in strengths:
        if s in (0.0, 1.0):
            continue
        group[float(s)] = interp(i0, i1, s, beta_scale=beta_scale) if use_retinex else interp(i0, i1, s)
    return group
