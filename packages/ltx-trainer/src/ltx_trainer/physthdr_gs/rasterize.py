"""Perspective HDR Gaussian rasterization (differentiable alpha splat, Sec. 3.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.physthdr_gs.cameras import Camera


def project_gaussians(
    xyz: Tensor,
    scales: Tensor,
    opacity: Tensor,
    cam: Camera,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project Gaussians; return uv, depth, opacity, radii, visible mask."""
    p_cam = cam.world_to_cam(xyz)
    uv, depth = cam.project(p_cam)
    h, w = cam.height, cam.width
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (depth > 0.1)
    op = opacity.squeeze(-1) if opacity.dim() > 1 else opacity
    mean_scale = scales.mean(dim=-1) if scales.dim() > 1 else scales
    radii = (mean_scale * cam.fx / depth.clamp(min=0.1)).clamp(min=1.0, max=64.0)
    return uv, depth, op, radii * inside.float(), inside


def render_perspective_hdr(
    xyz: Tensor,
    scales: Tensor,
    opacity: Tensor,
    colors: Tensor,
    cam: Camera,
    *,
    max_gaussians: int = 1024,
    fast: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Render HDR image ``[1, 3, H, W]`` and per-Gaussian screen-space gradient weights.

    Returns:
        image, viewspace_points (xyz with grad hook target for densification)
    """
    viewspace = xyz
    if fast:
        img = _render_fast(xyz, scales, opacity, colors, cam, max_gaussians=max_gaussians)
    else:
        img = _render_sorted(xyz, scales, opacity, colors, cam, max_gaussians=max_gaussians)
    return img.unsqueeze(0) if img.dim() == 3 else img, viewspace


def _select_visible(
    xyz: Tensor,
    scales: Tensor,
    opacity: Tensor,
    colors: Tensor,
    cam: Camera,
    max_gaussians: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    uv, depth, op, radii, inside = project_gaussians(xyz, scales, opacity, cam)
    valid = (op > 1e-3) & (radii > 0.5) & inside
    if not valid.any():
        empty = xyz[:0]
        return empty, empty, empty, empty, empty, uv
    idx = torch.where(valid)[0]
    if idx.numel() > max_gaussians:
        _, order = depth[idx].sort(descending=True)
        idx = idx[order[:max_gaussians]]
    return xyz[idx], scales[idx], op[idx], colors[idx], radii[idx], uv[idx]


def _render_sorted(
    xyz: Tensor,
    scales: Tensor,
    opacity: Tensor,
    colors: Tensor,
    cam: Camera,
    *,
    max_gaussians: int,
) -> Tensor:
    device = xyz.device
    h, w = cam.height, cam.width
    xyz, _, op, cols, radii, uv = _select_visible(xyz, scales, opacity, colors, cam, max_gaussians)
    if xyz.numel() == 0:
        return torch.zeros(3, h, w, device=device)

    depth = cam.world_to_cam(xyz)[:, 2]
    order = torch.argsort(depth, descending=True)
    uv = uv[order]
    cols = cols[order]
    op = op[order]
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
        alpha = (op[i] * g).clamp(0.0, 0.99)
        weight = alpha * trans
        acc += cols[i].view(3, 1, 1) * weight.unsqueeze(0)
        trans = trans * (1.0 - alpha)
        if trans.max() < 1e-3:
            break
    return acc.clamp(min=0.0)


def _render_fast(
    xyz: Tensor,
    scales: Tensor,
    opacity: Tensor,
    colors: Tensor,
    cam: Camera,
    *,
    max_gaussians: int,
) -> Tensor:
    device = xyz.device
    h, w = cam.height, cam.width
    xyz, _, op, cols, radii, uv = _select_visible(xyz, scales, opacity, colors, cam, max_gaussians)
    if xyz.numel() == 0:
        return torch.zeros(3, h, w, device=device)

    uv_n = torch.stack([2.0 * uv[:, 0] / w - 1.0, 2.0 * uv[:, 1] / h - 1.0], dim=-1)
    op_v = op.view(-1, 1, 1, 1)
    layers: list[Tensor] = []
    for i in range(uv_n.shape[0]):
        grid = uv_n[i].view(1, 1, 1, 2)
        feat = cols[i].view(1, 3, 1, 1).expand(1, 3, h, w)
        splat = F.grid_sample(feat, grid.expand(1, h, w, 2), mode="bilinear", align_corners=True)
        layers.append(splat.squeeze(0) * op_v[i])
    if not layers:
        return torch.zeros(3, h, w, device=device)
    return torch.stack(layers, dim=0).sum(dim=0).clamp(min=0.0)


def gaussian_blur(img: Tensor, kernel_size: int = 5) -> Tensor:
    """Gaussian blur for HDR consistency loss (Eq. 16)."""
    if kernel_size <= 1:
        return img
    pad = kernel_size // 2
    x = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - pad
    g = torch.exp(-0.5 * (x / max(pad, 1)) ** 2)
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1).expand(3, 1, 1, -1)
    kv = g.view(1, 1, -1, 1).expand(3, 1, -1, 1)
    img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    img = F.conv2d(img, kh, groups=3)
    img = F.conv2d(img, kv, groups=3)
    return img
