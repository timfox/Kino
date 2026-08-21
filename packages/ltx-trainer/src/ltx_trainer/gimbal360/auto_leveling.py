"""Differentiable Auto-Leveling: dense flow + soft-argmin rigid 3-DOF (Sec. 3.3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def soft_argmin_rigid_rotation(
    flow: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """
    Collapse dense 2D flow into a proxy 3×3 rotation via weighted Procrustes stub.
    flow: [B, 2, H, W] pixel displacements.
    """
    b, _, h, w = flow.shape
    device = flow.device
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    src = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    dst = src + flow * 0.1
    mag = flow.norm(dim=1)
    weights = torch.softmax((-mag / temperature).reshape(b, -1), dim=1).reshape(b, h, w)
    weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    # 2D rotation angle from mean flow (pitch/roll proxy)
    mean_dx = (flow[:, 0] * weights).sum(dim=(1, 2))
    mean_dy = (flow[:, 1] * weights).sum(dim=(1, 2))
    angle = torch.atan2(mean_dy, mean_dx + 1e-8)
    c, s = torch.cos(angle), torch.sin(angle)
    r00 = c
    r01 = -s
    r10 = s
    r11 = c
    r = torch.zeros(b, 3, 3, device=device)
    r[:, 0, 0] = r00
    r[:, 0, 1] = r01
    r[:, 1, 0] = r10
    r[:, 1, 1] = r11
    r[:, 2, 2] = 1.0
    return r


def warp_latent_spherical(z: Tensor, rotation: Tensor) -> Tensor:
    """Differentiable grid sample with rotation acting on latent plane (stub)."""
    b, c, h, w = z.shape
    theta = rotation[:, :2, :2]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=z.device),
        torch.linspace(-1, 1, w, device=z.device),
        indexing="ij",
    )
    coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    flat = coords.view(b, -1, 2)
    rot2 = theta.transpose(1, 2)
    warped = torch.bmm(flat, rot2) + rotation[:, :2, 2:3].transpose(1, 2)
    warped = warped.view(b, h, w, 2)
    return F.grid_sample(z, warped, mode="bilinear", padding_mode="border", align_corners=True)


class DifferentiableAutoLeveling(nn.Module):
    """SegNeXt-style stub: dense flow → soft-argmin → canonical warp."""

    def __init__(self, in_ch: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.flow_head = nn.Conv2d(hidden, 2, 1)

    def forward(self, perspective: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        feat = self.encoder(perspective)
        flow = self.flow_head(feat)
        rot = soft_argmin_rigid_rotation(flow)
        lh = max(perspective.shape[2] // 4, 4)
        lw = max(perspective.shape[3] // 4, 4)
        z_stub = F.adaptive_avg_pool2d(feat, (lh, lw))
        if z_stub.shape[1] < 4:
            z_stub = z_stub.repeat(1, (4 + z_stub.shape[1] - 1) // z_stub.shape[1], 1, 1)[:, :4]
        else:
            z_stub = z_stub[:, :4]
        z_canonical = warp_latent_spherical(z_stub, rot)
        return flow, rot, z_canonical
