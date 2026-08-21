"""Multilayer geometry representation utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def forward_fill_layers(x: Tensor, valid: Tensor) -> Tensor:
    """Forward-fill empty ray intersections along the layer axis (Eq. 2).

    Args:
        x: (L, H, W, 3) or (B, L, H, W, 3) camera-space points.
        valid: (L, H, W) bool — True where a true intersection exists.
    """
    batched = x.ndim == 5
    if not batched:
        x = x.unsqueeze(0)
        valid = valid.unsqueeze(0)
    b, layers, h, w, _ = x.shape
    out = x.clone()
    for bidx in range(b):
        for u in range(h):
            for v in range(w):
                last_valid = None
                for ell in range(layers):
                    if valid[bidx, ell, u, v]:
                        last_valid = out[bidx, ell, u, v].clone()
                    elif last_valid is not None:
                        out[bidx, ell, u, v] = last_valid
    return out if batched else out.squeeze(0)


def object_zscore_normalize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (x - mean.view(1, 1, 1, 3)) / std.view(1, 1, 1, 3).clamp_min(1e-6)


def object_zscore_denormalize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return x * std.view(1, 1, 1, 3) + mean.view(1, 1, 1, 3)


def scene_log_median_normalize(x: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
    """Per-sample log-median normalization for scenes (App. A Eq. 6)."""
    z = x[..., 2]
    mask = alpha > 0.5
    if mask.any():
        m = z[mask].median()
    else:
        m = z.abs().median().clamp_min(1e-3)
    m = m.clamp_min(1e-3)
    out = x.clone()
    out[..., 2] = torch.log(z / m)
    out[..., 0] = torch.sign(x[..., 0]) * torch.log1p(x[..., 0].abs() / m)
    out[..., 1] = torch.sign(x[..., 1]) * torch.log1p(x[..., 1].abs() / m)
    return out, m


def scene_log_median_denormalize(x: Tensor, m: Tensor) -> Tensor:
    out = x.clone()
    out[..., 2] = torch.exp(x[..., 2]) * m
    out[..., 0] = torch.sign(x[..., 0]) * (torch.expm1(x[..., 0].abs()) * m)
    out[..., 1] = torch.sign(x[..., 1]) * (torch.expm1(x[..., 1].abs()) * m)
    return out


def mix_training_loss_mask(alpha: Tensor, *, b_single: bool, num_layers: int) -> Tensor:
    """Gate endpoint loss along layers for RGBD-only samples (Eq. 4)."""
    b = alpha.shape[0]
    h, w = alpha.shape[-2], alpha.shape[-1]
    base = alpha.view(b, 1, h, w, 1).expand(b, num_layers, h, w, 3)
    if b_single:
        gate = torch.zeros(num_layers, device=alpha.device, dtype=alpha.dtype)
        gate[0] = 1.0
        return base * gate.view(1, num_layers, 1, 1, 1)
    return base


def invalid_pixel_noise_fill(xt: Tensor, alpha: Tensor) -> Tensor:
    """Replace invalid pixels with max Gaussian noise (Eq. 7)."""
    noise = torch.randn_like(xt)
    valid = alpha > 0.5
    if valid.ndim + 2 == xt.ndim:
        valid = valid.unsqueeze(1).unsqueeze(-1).expand_as(xt)
    else:
        while valid.ndim < xt.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.expand_as(xt)
    return torch.where(valid, xt, noise)
