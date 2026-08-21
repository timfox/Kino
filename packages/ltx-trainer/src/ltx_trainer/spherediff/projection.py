"""Spherical-to-perspective projection stub (Eq. 11–12, Sec. A.1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def view_rotation(azimuth_deg: float, elevation_deg: float) -> Tensor:
    """3×3 rotation: camera looks along view on sphere."""
    th = math.radians(azimuth_deg)
    ph = math.radians(elevation_deg)
    # forward = (sin(th)cos(ph), sin(ph), cos(th)cos(ph)) in x-right, y-down, z-forward
    fz = math.cos(th) * math.cos(ph)
    fx = math.sin(th) * math.cos(ph)
    fy = math.sin(ph)
    forward = torch.tensor([fx, fy, fz])
    forward = forward / forward.norm()
    up = torch.tensor([0.0, -1.0, 0.0])
    right = torch.linalg.cross(forward, up)
    right = right / right.norm().clamp_min(1e-6)
    up_c = torch.linalg.cross(right, forward)
    r = torch.stack([right, up_c, forward], dim=0)
    return r


def project_to_perspective(
    dirs: Tensor,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    fov_deg: float,
) -> tuple[Tensor, Tensor]:
    """
    Project unit directions to normalized perspective coords u ∈ [-1,1]².
    Returns (u, valid_mask).
    """
    r = view_rotation(azimuth_deg, elevation_deg)
    cam = dirs @ r.T
    z = cam[..., 2]
    valid = z > 1e-4
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    u = cam[..., 0] / z.clamp_min(1e-4) / f
    v = cam[..., 1] / z.clamp_min(1e-4) / f
    uv = torch.stack([u, v], dim=-1)
    in_fov = (uv.abs() <= 1.0).all(dim=-1) & valid
    return uv, in_fov
