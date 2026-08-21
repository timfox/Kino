"""Synthetic top-bottom ERP stereo pairs for smoke tests."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.h_omnistereo.config import HOmniStereoConfig
from ltx_trainer.h_omnistereo.heading_normal import camera_to_heading_aligned, longitude_grid
from ltx_trainer.h_omnistereo.spherical_geo import erp_row_grid, spherical_disparity


def synthetic_erp_pair(
    cfg: HOmniStereoConfig | None = None,
    *,
    batch: int = 1,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """Random RGB top/bottom + GT disparity and heading-aligned normals."""
    cfg = cfg or HOmniStereoConfig()
    h, w = cfg.train_crop_h, cfg.train_crop_w
    dev = device or torch.device("cpu")
    top = torch.rand(batch, 3, h, w, device=dev)
    bottom = top + 0.05 * torch.randn(batch, 3, h, w, device=dev)
    theta = erp_row_grid(h, w, device=dev)
    theta_top = theta * 0.45
    theta_bot = theta * 0.55 + 0.02
    rb = torch.full((h, w), 3.0, device=dev) + 0.5 * torch.sin(theta * 2)
    disp = spherical_disparity(theta_top, theta_bot, rb, cfg.baseline_m)
    disp = disp.unsqueeze(0).expand(batch, -1, -1)
    # Simple slanted plane normal in camera frame
    u = torch.linspace(-1, 1, w, device=dev)
    v = torch.linspace(-1, 1, h, device=dev)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    n_cam = torch.stack([uu * 0.3, vv * 0.3, torch.ones_like(uu)], dim=0)
    n_cam = torch.nn.functional.normalize(n_cam, dim=0)
    alpha = longitude_grid(w, h, device=dev)
    n_ha = camera_to_heading_aligned(n_cam.unsqueeze(0), alpha.unsqueeze(0))
    return {
        "top": top.clamp(0, 1),
        "bottom": bottom.clamp(0, 1),
        "disparity": disp.unsqueeze(1),
        "normal_ha": n_ha.expand(batch, -1, -1, -1),
        "normal_cam": n_cam.unsqueeze(0).expand(batch, -1, -1, -1),
    }
