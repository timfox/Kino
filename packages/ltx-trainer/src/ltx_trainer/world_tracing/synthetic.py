"""Synthetic multilayer RGBA + XYZ targets for CPU smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.world_tracing.config import WTConfig
from ltx_trainer.world_tracing.representation import forward_fill_layers


def synthetic_rgba_batch(cfg: WTConfig, *, batch: int = 1, device=None) -> dict[str, Tensor]:
    h, w = cfg.height, cfg.width
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    rgb = torch.stack([xx, yy, 0.5 * torch.ones_like(xx)], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    alpha = (xx.abs() < 0.8).float().unsqueeze(0).expand(batch, h, w)
    rgba = torch.cat([rgb, alpha.unsqueeze(1)], dim=1)

    layers = cfg.num_layers
    stack = []
    valid = []
    for ell in range(layers):
        z = 1.2 + 0.15 * ell + 0.05 * (xx**2 + yy**2)
        pts = torch.stack([xx * z, yy * z, z], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        stack.append(pts)
        v = (alpha > 0.5).unsqueeze(1).expand(batch, 1, h, w).squeeze(1)
        if ell > 2:
            v = v & (torch.rand_like(v.float()) > 0.3)
        valid.append(v)
    x = torch.stack(stack, dim=1)
    valid_t = torch.stack(valid, dim=1)
    filled = forward_fill_layers(x[0], valid_t[0]).unsqueeze(0).expand(batch, -1, -1, -1, -1)
    return {
        "rgba": rgba,
        "xyz": filled,
        "alpha": alpha,
        "valid_layers": valid_t,
    }
