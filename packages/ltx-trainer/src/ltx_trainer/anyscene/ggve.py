"""Geometry-Grounded View Expansion (Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.anyscene.config import AnySceneConfig

# Surround camera order (nuScenes-style)
SURROUND_VIEWS = ("FL", "F", "FR", "BL", "B", "BR")


def plucker_embedding(
    rays_o: Tensor,
    rays_d: Tensor,
) -> Tensor:
    """Plücker coordinates (6,) per ray: [d, o×d]."""
    cross = torch.cross(rays_o, rays_d, dim=-1)
    return torch.cat([rays_d, cross], dim=-1)


def pinhole_rays(
    intrinsics: Tensor,
    c2w: Tensor,
    h: int,
    w: int,
) -> tuple[Tensor, Tensor]:
    """Unit directions and origins for a pinhole camera."""
    device = intrinsics.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)
    k_inv = torch.inverse(intrinsics)
    dirs_cam = (pix @ k_inv.T).reshape(-1, 3)
    dirs_cam = F.normalize(dirs_cam, dim=-1)
    r = c2w[:3, :3]
    t = c2w[:3, 3]
    dirs_w = (dirs_cam @ r.T).reshape(h, w, 3)
    origins = t.view(1, 1, 3).expand(h, w, 3)
    return origins, dirs_w


def render_semantic_buffer(labels: Tensor, cam_height: int, cam_width: int) -> Tensor:
    """BEV occupancy (H,W,Z) → coarse semantic image (1, Hc, Wc)."""
    bev = labels.float().amax(dim=-1)
    if bev.dim() == 2:
        bev = bev.unsqueeze(0).unsqueeze(0)
    elif bev.dim() == 3:
        bev = bev.unsqueeze(1)
    return F.interpolate(bev, size=(cam_height, cam_width), mode="nearest")


def render_coordinate_buffer(
    labels: Tensor,
    cam_height: int,
    cam_width: int,
) -> Tensor:
    """Proxy coordinate buffer: stacked normalized grid + height index."""
    h, w = labels.shape[-3], labels.shape[-2]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=labels.device),
        torch.linspace(-1, 1, w, device=labels.device),
        indexing="ij",
    )
    height_idx = labels.float().argmax(dim=-1) / max(labels.shape[-1], 1)
    coord = torch.stack([xx, yy, height_idx], dim=0)
    if labels.dim() == 4:
        coord = coord.unsqueeze(0)
    else:
        coord = coord.unsqueeze(0)
    return F.interpolate(coord, size=(cam_height, cam_width), mode="bilinear", align_corners=False)


class GGVEControlHints(nn.Module):
    """VACE-style control hints: RGB latent + Plücker + semantic/coord buffers."""

    def __init__(self, cfg: AnySceneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = 64
        self.sem_embed = nn.Conv2d(1, d, 3, padding=1)
        self.coord_embed = nn.Conv2d(3, d, 3, padding=1)
        self.plucker_embed = nn.Conv2d(cfg.plucker_dim, d, 3, padding=1)
        self.fuse = nn.Conv2d(d * 3, d, 1)

    def forward(
        self,
        semantic: Tensor,
        coord: Tensor,
        plucker: Tensor,
    ) -> Tensor:
        h = self.fuse(
            torch.cat(
                [
                    self.sem_embed(semantic),
                    self.coord_embed(coord),
                    self.plucker_embed(plucker.permute(0, 3, 1, 2)),
                ],
                dim=1,
            )
        )
        return h


def surround_expansion_schedule() -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Four-step autoregressive surround (supplement B.3)."""
    return [
        ("step1", ("FL", "F", "FR"), ()),
        ("step2", ("FL",), ("BL",)),
        ("step3", ("FR",), ("BR",)),
        ("step4", ("BL", "BR"), ("B",)),
    ]
