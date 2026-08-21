"""Dense auxiliary labels (supp. A.3): gradient, EDF, metric point map."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.mtpano.erp import erp_spherical_coords, spherical_ray_direction

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)


def image_gradient_map(image: Tensor) -> Tensor:
    """Sobel magnitude + direction (Eq. 7), shape (B, 2, H, W)."""
    dev, dtype = image.device, image.dtype
    gray = image.mean(dim=1, keepdim=True)
    kx = _SOBEL_X.to(device=dev, dtype=dtype)
    ky = _SOBEL_Y.to(device=dev, dtype=dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    ang = torch.atan2(gy, gx)
    return torch.cat([mag, ang], dim=1)


def jump_flood_edf(edge_mask: Tensor, *, max_steps: int | None = None) -> Tensor:
    """Edge distance field via Jump Flooding (Rong & Tan 2006), (B, 1, H, W)."""
    if edge_mask.dim() == 3:
        edge_mask = edge_mask.unsqueeze(1)
    b, _, h, w = edge_mask.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=edge_mask.device, dtype=torch.float32),
        torch.arange(w, device=edge_mask.device, dtype=torch.float32),
        indexing="ij",
    )
    base = torch.stack([yy, xx], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    is_edge = edge_mask.squeeze(1) > 0.5
    nearest = base.clone()
    inf = float(h + w + 1)
    nearest[~is_edge.unsqueeze(-1).expand_as(nearest)] = inf

    step = 1 << (max(h, w) - 1).bit_length()
    if max_steps is not None:
        step = min(step, 1 << max_steps)
    while step >= 1:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                shifted = torch.roll(nearest, shifts=(dy, dx), dims=(1, 2))
                dist_cur = (base - nearest).pow(2).sum(-1)
                dist_new = (base - shifted).pow(2).sum(-1)
                use_new = dist_new < dist_cur
                nearest = torch.where(use_new.unsqueeze(-1), shifted, nearest)
        step //= 2

    dist = torch.sqrt((base - nearest).pow(2).sum(-1) + 1e-8)
    return dist.unsqueeze(1)


def edge_distance_field(image: Tensor, *, edge_tau: float = 0.99) -> Tensor:
    """EDF from gradient edges (supp. A.3)."""
    grad = image_gradient_map(image)
    mag = grad[:, :1]
    peak = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    edges = (mag >= edge_tau * peak).float()
    edges[:, :, 0, :] = 0.0
    edges[:, :, -1, :] = 0.0
    edges[:, :, :, 0] = 0.0
    edges[:, :, :, -1] = 0.0
    return jump_flood_edf(edges)


def metric_point_map(depth: Tensor, height: int, width: int) -> Tensor:
    """P = D ⊙ r with ERP rays (Eq. 8), depth (B, 1, H, W)."""
    phi, theta = erp_spherical_coords(height, width, device=depth.device, dtype=depth.dtype)
    r = spherical_ray_direction(phi, theta)  # H, W, 3
    r = r.permute(2, 0, 1).unsqueeze(0)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    return depth * r


def build_auxiliary_targets(image: Tensor, depth: Tensor | None = None) -> dict[str, Tensor]:
    """Gradient + EDF (+ point map when depth provided)."""
    out: dict[str, Tensor] = {
        "grad": image_gradient_map(image),
        "edf": edge_distance_field(image),
    }
    if depth is not None:
        _, _, h, w = image.shape
        out["point_map"] = metric_point_map(depth, h, w)
    return out
