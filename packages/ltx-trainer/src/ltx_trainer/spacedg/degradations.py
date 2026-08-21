"""Physically grounded degradations (Appendix B, Table 9)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.spacedg.schema import DegradationType

DEGRADATION_PARAM_RANGES: dict[str, str] = {
    "defocus": "aperture ∈ [10.0, 15.0], depth ∈ [1.0, 8.0]",
    "distortion": "k1 ∈ [-0.24, -0.23], k2 ∈ [0.0001, 0.0003]",
    "haze": "density ∈ [3.5, 6.0]",
    "jpeg_compression": "quality ∈ [2, 5]",
    "low_light": "exposure ∈ [0.003, 0.005]",
    "low_resolution": "scale ∈ [0.02, 0.05]",
    "motion_blur": "trans ∈ [0.2, 0.35], rot ∈ [0.06, 0.12]",
    "over_exposure": "exposure ∈ [7.0, 10.0]",
    "water_droplets": "scale ∈ [2.5, 4.0], strength ∈ [0.3, 0.5]",
}


@dataclass
class DegradationParams:
    """Sampled parameters for one degradation instance."""

    kind: DegradationType
    values: dict[str, float]


def sample_params(kind: DegradationType) -> DegradationParams:
    if kind == DegradationType.ORIGINAL:
        return DegradationParams(kind, {})
    if kind == DegradationType.DEFOCUS:
        return DegradationParams(kind, {"aperture": random.uniform(10.0, 15.0), "depth": random.uniform(1.0, 8.0)})
    if kind == DegradationType.DISTORTION:
        return DegradationParams(kind, {"k1": random.uniform(-0.24, -0.23)})
    if kind == DegradationType.HAZE:
        return DegradationParams(kind, {"beta": random.uniform(3.5, 6.0) / 10.0})
    if kind == DegradationType.JPEG_COMPRESSION:
        return DegradationParams(kind, {"quality": random.uniform(2.0, 5.0) / 10.0})
    if kind == DegradationType.LOW_LIGHT:
        return DegradationParams(kind, {"exposure": random.uniform(0.003, 0.005)})
    if kind == DegradationType.LOW_RESOLUTION:
        return DegradationParams(kind, {"scale": random.uniform(0.02, 0.05)})
    if kind == DegradationType.MOTION_BLUR:
        return DegradationParams(kind, {"angle": random.uniform(0, 3.14159), "length": random.uniform(8, 16)})
    if kind == DegradationType.OVER_EXPOSURE:
        return DegradationParams(kind, {"exposure": random.uniform(7.0, 10.0)})
    if kind == DegradationType.WATER_DROPLETS:
        return DegradationParams(kind, {"strength": random.uniform(0.3, 0.5)})
    return DegradationParams(kind, {})


def apply_degradation(
    rgb: Tensor,
    kind: DegradationType | str,
    *,
    depth: Tensor | None = None,
    params: DegradationParams | None = None,
) -> Tensor:
    """Apply degradation to ``rgb [3,H,W]`` in ``[0,1]`` (linear or gamma)."""
    dt = DegradationType(kind) if not isinstance(kind, DegradationType) else kind
    if dt == DegradationType.ORIGINAL:
        return rgb.clamp(0.0, 1.0)
    p = params or sample_params(dt)
    if dt == DegradationType.DEFOCUS:
        sigma = p.values.get("aperture", 12.0) / p.values.get("depth", 4.0) * 0.15
        return _gaussian_blur(rgb, sigma)
    if dt == DegradationType.DISTORTION:
        return _barrel_distort(rgb, k1=p.values.get("k1", -0.23))
    if dt == DegradationType.HAZE:
        return _haze(rgb, depth, beta=p.values.get("beta", 0.5))
    if dt == DegradationType.JPEG_COMPRESSION:
        return _jpeg_proxy(rgb, quality=p.values.get("quality", 0.3))
    if dt == DegradationType.LOW_LIGHT:
        return _low_light(rgb, exposure=p.values.get("exposure", 0.004))
    if dt == DegradationType.LOW_RESOLUTION:
        return _low_res(rgb, scale=p.values.get("scale", 0.04))
    if dt == DegradationType.MOTION_BLUR:
        return _motion_blur(rgb, angle=p.values.get("angle", 0.0), length=p.values.get("length", 10.0))
    if dt == DegradationType.OVER_EXPOSURE:
        return _over_exposure(rgb, gain=p.values.get("exposure", 8.0))
    if dt == DegradationType.WATER_DROPLETS:
        return _water_droplets(rgb, strength=p.values.get("strength", 0.4))
    return rgb


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    k = max(3, int(sigma * 4) | 1)
    t = x.unsqueeze(0)
    t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, stride=1)
    return t.squeeze(0).clamp(0.0, 1.0)


def _barrel_distort(rgb: Tensor, *, k1: float) -> Tensor:
    c, h, w = rgb.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=rgb.device, dtype=rgb.dtype),
        torch.linspace(-1, 1, w, device=rgb.device, dtype=rgb.dtype),
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy).clamp(max=1.5)
    factor = 1.0 + k1 * r * r
    gx = (xx * factor).clamp(-1, 1)
    gy = (yy * factor).clamp(-1, 1)
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    out = F.grid_sample(rgb.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(0)


def _haze(rgb: Tensor, depth: Tensor | None, *, beta: float) -> Tensor:
    if depth is None:
        c, h, w = rgb.shape
        yy = torch.linspace(0, 1, h, device=rgb.device).view(h, 1).expand(h, w)
        depth = 1.0 - yy
    else:
        depth = depth.squeeze()
    t = torch.exp(-beta * depth.clamp(0, 1))
    A = rgb.mean(dim=(1, 2), keepdim=True)
    return (rgb * t + A * (1.0 - t)).clamp(0.0, 1.0)


def _jpeg_proxy(rgb: Tensor, *, quality: float) -> Tensor:
    down = max(2, int(8 * (1.0 - quality) + 2))
    small = F.interpolate(rgb.unsqueeze(0), scale_factor=1.0 / down, mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    return (back.squeeze(0) + torch.randn_like(rgb) * 0.02).clamp(0.0, 1.0)


def _low_light(rgb: Tensor, *, exposure: float) -> Tensor:
    x = rgb * exposure
    noise = torch.randn_like(x) * 0.05
    return (x + noise).clamp(0.0, 1.0)


def _low_res(rgb: Tensor, *, scale: float) -> Tensor:
    h, w = rgb.shape[-2:]
    sh = max(4, int(h * scale))
    sw = max(4, int(w * scale))
    small = F.interpolate(rgb.unsqueeze(0), size=(sh, sw), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


def _motion_blur(rgb: Tensor, *, angle: float, length: float) -> Tensor:
    k = max(3, int(length) | 1)
    kernel = torch.zeros(1, 1, k, k, device=rgb.device, dtype=rgb.dtype)
    cx, cy = k // 2, k // 2
    dx = int(torch.cos(torch.tensor(angle)) * (k // 2))
    dy = int(torch.sin(torch.tensor(angle)) * (k // 2))
    for t in range(-k // 2, k // 2 + 1):
        x = cx + int(dx * t / max(k // 2, 1))
        y = cy + int(dy * t / max(k // 2, 1))
        if 0 <= x < k and 0 <= y < k:
            kernel[0, 0, y, x] = 1.0
    kernel = kernel / kernel.sum().clamp(min=1e-6)
    out = F.conv2d(rgb.unsqueeze(0), kernel.expand(3, 1, k, k), padding=k // 2, groups=3)
    return out.squeeze(0).clamp(0.0, 1.0)


def _over_exposure(rgb: Tensor, *, gain: float) -> Tensor:
    return (rgb * gain / 10.0).clamp(0.0, 1.0)


def _water_droplets(rgb: Tensor, *, strength: float) -> Tensor:
    c, h, w = rgb.shape
    droplets = torch.rand(h, w, device=rgb.device) > (1.0 - strength * 0.1)
    blur = _gaussian_blur(rgb, 2.0)
    mask = droplets.float().unsqueeze(0)
    return (rgb * (1.0 - mask * 0.5) + blur * mask * 0.5).clamp(0.0, 1.0)
