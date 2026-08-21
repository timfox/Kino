"""Synthetic maritime clip for smoke tests (no real ASV logs required)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.traj_i2v.config import TrajI2VConfig
from ltx_trainer.traj_i2v.gps_mapping import GpsFix, VesselAnchor


def synthetic_reference_frame(
    cfg: TrajI2VConfig,
    *,
    seed: int = 0,
) -> Tensor:
    """Low-texture water field with two bright blobs (vessels)."""
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.image_height, cfg.image_width
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h),
        torch.linspace(0, 1, w),
        indexing="ij",
    )
    water = torch.stack([xx * 0.15 + 0.1, xx * 0.25 + 0.2, 0.45 + yy * 0.1], dim=0)
    for cx, cy, col in ((0.35, 0.45, (0.9, 0.9, 0.95)), (0.62, 0.52, (0.95, 0.9, 0.2))):
        dist = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
        blob = torch.exp(-dist * 80)
        for c in range(3):
            water[c] = water[c] + blob * col[c]
    return water.clamp(0, 1)


def synthetic_gps_log(
    cfg: TrajI2VConfig,
    *,
    t_start: float = 0.0,
) -> dict[int, list[GpsFix]]:
    """Two vessels moving east-north at ~1 Hz sample rate."""
    fixes: dict[int, list[GpsFix]] = {99999: [], 100000: []}
    origin_lon, origin_lat = 24.0, 37.5
    for i in range(cfg.num_frames + 4):
        t = t_start + i / cfg.fps
        fixes[99999].append(
            GpsFix(t_s=t, lon=origin_lon + 0.00012 * i, lat=origin_lat + 0.00001 * i, vessel_id=99999)
        )
        fixes[100000].append(
            GpsFix(
                t_s=t,
                lon=origin_lon + 0.00010 * i + 0.00005,
                lat=origin_lat + 0.00002 * i,
                vessel_id=100000,
            )
        )
    return fixes


def synthetic_anchors(cfg: TrajI2VConfig) -> list[VesselAnchor]:
    w, h = cfg.image_width, cfg.image_height
    return [
        VesselAnchor(99999, w * 0.35, h * 0.45, 24.0, 37.5, 0.0),
        VesselAnchor(100000, w * 0.62, h * 0.52, 24.00005, 37.50002, 0.0),
    ]
