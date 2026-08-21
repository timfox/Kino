"""Training / eval demo steps."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from ltx_trainer.panoworld_x.config import PanoWorldXConfig
from ltx_trainer.panoworld_x.erp_geometry import patch_spherical_coords, spherical_distance_haversine
from ltx_trainer.panoworld_x.losses import reconstruction_loss, ssim_stub
from ltx_trainer.panoworld_x.panoworld_x_net import PanoWorldXStub
from ltx_trainer.panoworld_x.route_sampling import pipeline_report
from ltx_trainer.panoworld_x.synthetic import synthetic_erp_frame, synthetic_route


def evaluation_demo_run(cfg: PanoWorldXConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PanoWorldXConfig(height=96, width=192, patch_size=16, num_frames=8)
    model = PanoWorldXStub(cfg, num_blocks=1)
    frame = synthetic_erp_frame(cfg)
    route = synthetic_route(cfg)
    out = model(frame, route)
    loss = reconstruction_loss(out["reconstruction"], frame)
    return {
        "frame_shape": list(frame.shape),
        "route_shape": list(route.shape),
        "recon_shape": list(out["reconstruction"].shape),
        "mse": float(loss.detach()),
        "panoexplorer_pipeline": pipeline_report(),
    }


def ablation_branches(cfg: PanoWorldXConfig | None = None) -> dict[str, float]:
    cfg = cfg or PanoWorldXConfig(height=96, width=192, patch_size=16)
    frame = synthetic_erp_frame(cfg)
    route = synthetic_route(cfg)
    full = PanoWorldXStub(cfg, num_blocks=1)
    no_sphere = PanoWorldXStub(replace(cfg, use_sphere_branch=False), num_blocks=1)
    no_exp = PanoWorldXStub(replace(cfg, use_exp_branch=False), num_blocks=1)
    with torch.no_grad():
        lf = float(reconstruction_loss(full(frame, route)["reconstruction"], frame))
        ls = float(reconstruction_loss(no_sphere(frame, route)["reconstruction"], frame))
        le = float(reconstruction_loss(no_exp(frame, None)["reconstruction"], frame))
    return {"full_mse": lf, "no_sphere_mse": ls, "no_exp_mse": le}


def sphere_edge_connectivity(cfg: PanoWorldXConfig | None = None) -> dict[str, float]:
    """Left/right ERP seam: spherical distance vs Euclidean in pixel space."""
    cfg = cfg or PanoWorldXConfig()
    theta, phi = patch_spherical_coords(cfg.height, cfg.width, cfg.patch_size)
    left = 0
    right = (cfg.width // cfg.patch_size) - 1
    gh = cfg.height // cfg.patch_size
    mid = gh // 2
    idx_left = mid * (cfg.width // cfg.patch_size) + left
    idx_right = mid * (cfg.width // cfg.patch_size) + right
    d_sphere = float(
        spherical_distance_haversine(theta[idx_left], phi[idx_left], theta[idx_right], phi[idx_right])
    )
    d_eucl = float(torch.sqrt((theta[idx_left] - theta[idx_right]) ** 2 + (phi[idx_left] - phi[idx_right]) ** 2))
    return {"spherical_seam_distance": d_sphere, "latlon_euclidean_seam": d_eucl}


def train_step(cfg: PanoWorldXConfig | None = None) -> dict[str, float]:
    cfg = cfg or PanoWorldXConfig(height=64, width=128, patch_size=16, num_frames=4)
    model = PanoWorldXStub(cfg, num_blocks=1)
    frame = synthetic_erp_frame(cfg)
    route = synthetic_route(cfg)
    out = model(frame, route)
    loss = reconstruction_loss(out["reconstruction"], frame) + 0.1 * ssim_stub(out["reconstruction"], frame)
    loss.backward()
    return {"loss": float(loss.detach())}
