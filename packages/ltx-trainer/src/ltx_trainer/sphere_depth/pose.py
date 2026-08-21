"""ERP pose perturbation via 3D ray rotation (Sec. 3.3)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _rotation_matrix_pitch_roll(pitch_deg: float, roll_deg: float, device: torch.device) -> Tensor:
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    # R = R_roll @ R_pitch (camera-centric)
    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        device=device,
        dtype=torch.float32,
    )
    ry = torch.tensor(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        device=device,
        dtype=torch.float32,
    )
    return rx @ ry


def erp_pixel_grid(height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
    v = torch.arange(height, device=device, dtype=torch.float32)
    u = torch.arange(width, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return uu, vv


def pixels_to_rays(u: Tensor, v: Tensor, height: int, width: int) -> Tensor:
    lam = 2.0 * math.pi * (u / width - 0.5)
    phi = math.pi * (0.5 - v / height)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    sin_lam = torch.sin(lam)
    cos_lam = torch.cos(lam)
    x = cos_phi * sin_lam
    y = sin_phi
    z = cos_phi * cos_lam
    return torch.stack([x, y, z], dim=-1)


def rays_to_pixels(rays: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    lam = torch.atan2(x, z)
    phi = torch.asin(y.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    u = (lam / (2.0 * math.pi) + 0.5) * width
    v = (0.5 - phi / math.pi) * height
    return u, v


def apply_pose_perturbation(
    erp: Tensor,
    pitch_deg: float,
    roll_deg: float,
) -> Tensor:
    """
    Rotate viewing directions by pitch/roll and resample ERP (B×C×H×W).
    """
    b, c, h, w = erp.shape
    device = erp.device
    u, v = erp_pixel_grid(h, w, device)
    rays = pixels_to_rays(u, v, h, w)
    r_mat = _rotation_matrix_pitch_roll(pitch_deg, roll_deg, device)
    rays_rot = rays @ r_mat.T
    u_src, v_src = rays_to_pixels(rays_rot, h, w)
    # grid_sample expects normalized [-1, 1]
    gx = (u_src / (w - 1)) * 2.0 - 1.0
    gy = (v_src / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(erp, grid, mode="bilinear", padding_mode="border", align_corners=True)


def pose_error_grid(
    depth_fn,
    erp: Tensor,
    gt_at_pose: dict[tuple[float, float], Tensor],
    pitch_values: tuple[float, ...],
    roll_values: tuple[float, ...],
    lambda_scale: float,
) -> Tensor:
    """
    Build |pitch|×|roll| heatmap of landmark MSE for sensitivity plots (Fig. 3).
    depth_fn: (erp_tensor) -> landmark depth predictions [L]
    """
    rows = len(pitch_values)
    cols = len(roll_values)
    grid = torch.zeros(rows, cols)
    for i, pitch in enumerate(pitch_values):
        for j, roll in enumerate(roll_values):
            perturbed = apply_pose_perturbation(erp, pitch, roll)
            pred = depth_fn(perturbed)
            key = (pitch, roll)
            gt = gt_at_pose.get(key, gt_at_pose.get((0.0, 0.0)))
            if gt is not None:
                grid[i, j] = landmark_mse_scalar(pred, gt, lambda_scale)
    return grid


def landmark_mse_scalar(pred: Tensor, gt: Tensor, lambda_scale: float) -> float:
    return float(((lambda_scale * pred.float() - gt.float()) ** 2).mean().item())
