"""Multi-view cameras for PhysHDR-GS (SfM export or synthetic rig)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.p2gs.cameras import Camera, init_points_from_ply, load_image_chw, load_scene_cameras

__all__ = [
    "Camera",
    "camera_to_device",
    "init_points_from_ply",
    "load_image_chw",
    "load_scene_cameras",
    "scale_cameras",
    "synthetic_camera_rig",
]


def camera_to_device(cam: Camera, device: torch.device | str) -> Camera:
    """Return a camera with pose tensors on ``device`` (for CUDA rasterization)."""
    dev = torch.device(device)
    return Camera(
        cam.image_path,
        cam.width,
        cam.height,
        cam.fx,
        cam.fy,
        cam.cx,
        cam.cy,
        cam.R.to(dev),
        cam.t.to(dev),
    )


def scale_cameras(cameras: list[Camera], scale: float) -> list[Camera]:
    if scale == 1.0:
        return cameras
    out: list[Camera] = []
    for c in cameras:
        out.append(
            Camera(
                c.image_path,
                max(1, int(c.width * scale)),
                max(1, int(c.height * scale)),
                c.fx * scale,
                c.fy * scale,
                c.cx * scale,
                c.cy * scale,
                c.R,
                c.t,
            )
        )
    return out


def synthetic_camera_rig(
    num_views: int,
    width: int,
    height: int,
    *,
    radius: float = 4.0,
    fov_deg: float = 50.0,
    device: torch.device | str = "cpu",
) -> list[Camera]:
    """Place pinhole cameras on a ring (for training without COLMAP)."""
    device = torch.device(device)
    fx = fy = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    cameras: list[Camera] = []
    for i in range(num_views):
        theta = 2.0 * math.pi * i / num_views
        # Camera on +Z arc looking at origin
        eye = torch.tensor(
            [radius * math.sin(theta), 0.15 * math.sin(theta * 2), radius * math.cos(theta)],
            dtype=torch.float32,
            device=device,
        )
        target = torch.zeros(3, device=device)
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        z = (eye - target)
        z = z / z.norm()
        x = torch.linalg.cross(up, z)
        x = x / (x.norm() + 1e-8)
        y = torch.linalg.cross(z, x)
        R = torch.stack([x, y, z], dim=0)
        t = -R @ eye
        cameras.append(
            Camera(
                image_path=Path(f"synthetic_view_{i:03d}.png"),
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                R=R,
                t=t,
            )
        )
    return cameras
