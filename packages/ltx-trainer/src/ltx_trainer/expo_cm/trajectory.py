"""Exposure-Aware Consistency Trajectory (EACT, Eq. 8–11)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.expo_cm.exposure_mask import ExposureMaskConfig, compute_exposure_masks


@dataclass
class EACTConfig:
    t_max: float = 1.0
    sigma_max: float = 0.8
    sigma_over_scale: float = 1.25
    lambda_under: float = 0.5
    blur_kernel: int = 5
    blur_sigma: float = 2.0
    mask_cfg: ExposureMaskConfig | None = None


def _alpha(t: Tensor, t_max: float) -> Tensor:
    return (t / t_max).clamp(0.0, 1.0)


def _sigma(t: Tensor, *, t_max: float, sigma_max: float, scale: float = 1.0) -> Tensor:
    return (t / t_max * sigma_max * scale).clamp(min=0.0)


def low_pass(ldr: Tensor, *, kernel: int = 5, sigma: float = 2.0) -> Tensor:
    """Gaussian low-pass ``Flow(y0)`` for under-exposed trajectory (Eq. 9)."""
    if ldr.dim() == 3:
        ldr = ldr.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    b, c, h, w = ldr.shape
    coords = torch.arange(kernel, device=ldr.device, dtype=ldr.dtype) - kernel // 2
    g1 = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1 = g1 / g1.sum()
    g2 = g1.view(1, 1, -1, 1) * g1.view(1, 1, 1, -1)
    g2 = g2.expand(c, 1, kernel, kernel)
    pad = kernel // 2
    out = F.conv2d(ldr, g2, padding=pad, groups=c)
    return out.squeeze(0) if squeeze else out


def sample_eact_state(
    x0: Tensor,
    y0: Tensor,
    t: Tensor | float,
    noise: Tensor | None = None,
    *,
    cfg: EACTConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sample ``x_t`` on the blended EACT (Eq. 11). Returns ``(x_t, masks)``."""
    cfg = cfg or EACTConfig()
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
        y0 = y0.unsqueeze(0)
    b = x0.shape[0]
    if isinstance(t, (int, float)):
        t = torch.full((b,), float(t), device=x0.device, dtype=x0.dtype)
    elif t.ndim == 0:
        t = t.reshape(1).expand(b)
    t_v = t.view(b, 1, 1, 1)
    eps = noise if noise is not None else torch.randn_like(x0)
    a = _alpha(t_v, cfg.t_max)
    sg = _sigma(t_v, t_max=cfg.t_max, sigma_max=cfg.sigma_max)
    so = _sigma(t_v, t_max=cfg.t_max, sigma_max=cfg.sigma_max, scale=cfg.sigma_over_scale)
    su = _sigma(t_v, t_max=cfg.t_max, sigma_max=cfg.sigma_max)
    y_blur = low_pass(y0, kernel=cfg.blur_kernel, sigma=cfg.blur_sigma)
    x_over = (1.0 - a) * x0 + so * eps
    x_under = (1.0 - a) * x0 + a * cfg.lambda_under * y_blur + su * eps
    x_good = (1.0 - a) * x0 + a * y0 + sg * eps
    masks = compute_exposure_masks(y0, cfg.mask_cfg)
    wover = masks["wover"]
    wunder = masks["wunder"]
    wgood = masks["wgood"]
    if wover.dim() == 3:
        wover = wover.unsqueeze(0)
        wunder = wunder.unsqueeze(0)
        wgood = wgood.unsqueeze(0)
    xt = wover * x_over + wunder * x_under + wgood * x_good
    return xt, masks
