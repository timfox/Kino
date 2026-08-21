"""Base EV0 panorama from six orthogonal views (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_CUBEMAP_DIRS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def render_cubemap_faces(scene_rgb: Tensor, face_size: int) -> Tensor:
    """
    Stub: derive six face colors from a single scene tensor [B,3,H,W].
    Returns [B,6,3,face_size,face_size].
    """
    b, c, h, w = scene_rgb.shape
    faces = []
    for i in range(6):
        shift = (i + 1) * 0.03
        face = F.interpolate(scene_rgb, (face_size, face_size), mode="bilinear", align_corners=False)
        faces.append((face * (0.85 + 0.05 * i) + shift).clamp(0, 1))
    return torch.stack(faces, dim=1)


def cubemap_to_equirect(faces: Tensor, out_h: int, out_w: int) -> Tensor:
    """Stitch cubemap faces into equirectangular panorama (differentiable stub)."""
    b, _, c, fh, fw = faces.shape
    u = torch.linspace(0, 1, out_w, device=faces.device)
    v = torch.linspace(0, 1, out_h, device=faces.device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    lon = (uu - 0.5) * 2 * torch.pi
    lat = (vv - 0.5) * torch.pi
    cx = torch.cos(lat) * torch.sin(lon)
    cy = torch.sin(lat)
    cz = torch.cos(lat) * torch.cos(lon)
    ax, ay, az = cx.abs(), cy.abs(), cz.abs()
    face_idx = torch.zeros_like(ax, dtype=torch.long)
    face_idx = torch.where((ax >= ay) & (ax >= az), torch.where(cx > 0, 0, 1), face_idx)
    face_idx = torch.where((ay >= ax) & (ay >= az), torch.where(cy > 0, 2, 3), face_idx)
    face_idx = torch.where((az >= ax) & (az >= ay), torch.where(cz > 0, 4, 5), face_idx)

    out = torch.zeros(b, c, out_h, out_w, device=faces.device)
    for fi in range(6):
        mask = face_idx == fi
        if not mask.any():
            continue
        sampled = F.interpolate(faces[:, fi], (out_h, out_w), mode="bilinear", align_corners=False)
        out = out + sampled * mask.unsqueeze(0).unsqueeze(0).float()
    return out.clamp(0, 1)


def build_ev0_panorama(scene_rgb: Tensor, *, out_h: int = 32, out_w: int = 64) -> Tensor:
    face_size = min(scene_rgb.shape[-1], scene_rgb.shape[-2])
    faces = render_cubemap_faces(scene_rgb, max(8, face_size // 2))
    return cubemap_to_equirect(faces, out_h, out_w)
