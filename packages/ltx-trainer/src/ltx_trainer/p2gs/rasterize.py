"""Differentiable HDR splatting for P2GS (alpha compositing, Eq. 1–3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.p2gs.cameras import Camera
from ltx_trainer.p2gs.gaussians import GaussianModel


def _project_gaussians(
    model: GaussianModel,
    cam: Camera,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project 3D Gaussians to image plane; return uv, depth, colors, opacity, radii."""
    xyz = model.xyz
    p_cam = cam.world_to_cam(xyz)
    uv, depth = cam.project(p_cam)
    h, w = cam.height, cam.width
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (depth > 0.1)
    colors = model.linear_rgb
    opacity = model.opacity().squeeze(-1)
    # Screen-space radius from focal length and depth
    mean_scale = model.scales().mean(dim=-1)
    radii = (mean_scale * cam.fx / depth.clamp(min=0.1)).clamp(min=1.0, max=64.0)
    return uv, depth, colors, opacity, radii * inside.float()


def render_linear_hdr(
    model: GaussianModel,
    cam: Camera,
    *,
    max_gaussians: int = 512,
    tile_size: int = 8,
) -> Tensor:
    """
    Render scene-linear HDR ``[3, H, W]`` via sorted alpha splatting.

    Uses perspective projection and 2D Gaussian footprints (differentiable).
    """
    device = model.xyz.device
    h, w = cam.height, cam.width
    uv, depth, colors, opacity, radii = _project_gaussians(model, cam)
    valid = (opacity > 1e-3) & (radii > 0.5)
    if not valid.any():
        return torch.zeros(3, h, w, device=device)

    idx = torch.where(valid)[0]
    if idx.numel() > max_gaussians:
        _, order = depth[idx].sort(descending=True)
        idx = idx[order[:max_gaussians]]

    uv = uv[idx]
    depth = depth[idx]
    colors = colors[idx]
    opacity = opacity[idx]
    radii = radii[idx]

    # Back-to-front compositing
    order = torch.argsort(depth, descending=True)
    uv = uv[order]
    colors = colors[order]
    opacity = opacity[order]
    radii = radii[order]

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    acc = torch.zeros(3, h, w, device=device)
    trans = torch.ones(h, w, device=device)

    for i in range(uv.shape[0]):
        du = xx - uv[i, 0]
        dv = yy - uv[i, 1]
        g = torch.exp(-0.5 * (du**2 + dv**2) / (radii[i] ** 2 + 1e-4))
        alpha = (opacity[i] * g).clamp(0.0, 0.99)
        weight = alpha * trans
        acc += colors[i].view(3, 1, 1) * weight.unsqueeze(0)
        trans = trans * (1.0 - alpha)
        if trans.max() < 1e-3:
            break

    return acc.clamp(min=0.0)


def render_linear_hdr_fast(
    model: GaussianModel,
    cam: Camera,
    *,
    max_gaussians: int = 256,
) -> Tensor:
    """Grid-based soft splat (faster training iterations)."""
    device = model.xyz.device
    h, w = cam.height, cam.width
    uv, depth, colors, opacity, radii = _project_gaussians(model, cam)
    valid = (opacity > 1e-3) & (radii > 0.5)
    if not valid.any():
        return torch.zeros(3, h, w, device=device)

    idx = torch.where(valid)[0]
    if idx.numel() > max_gaussians:
        _, order = depth[idx].sort(descending=True)
        idx = idx[order[:max_gaussians]]

    uv_n = torch.stack([2.0 * uv[idx, 0] / w - 1.0, 2.0 * uv[idx, 1] / h - 1.0], dim=-1)
    sigma = (radii[idx] / max(h, w)).clamp(min=0.01, max=0.25).view(-1, 1, 1)
    cols = colors[idx]
    op = opacity[idx].view(-1, 1, 1, 1)

    layers = []
    for i in range(uv_n.shape[0]):
        grid = uv_n[i].view(1, 1, 1, 2)
        feat = cols[i].view(1, 3, 1, 1).expand(1, 3, h, w)
        splat = F.grid_sample(feat, grid.expand(1, h, w, 2), mode="bilinear", align_corners=True)
        layers.append(splat.squeeze(0) * op[i])

    if not layers:
        return torch.zeros(3, h, w, device=device)
    acc = torch.stack(layers, dim=0).sum(dim=0)
    return acc.clamp(min=0.0)
