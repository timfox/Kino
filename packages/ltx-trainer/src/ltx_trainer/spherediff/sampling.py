"""Dynamic latent sampling (Algorithm 1, Sec. 3.3)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.spherediff.projection import project_to_perspective


def nearest_sampling(uv: Tensor, feats: Tensor, h: int, w: int) -> Tensor:
    """Nearest-neighbor grid assignment (Fig. 4a baseline)."""
    grid = torch.zeros(h, w, feats.shape[-1], device=feats.device)
    for j in range(h):
        for k in range(w):
            u = -1.0 + 2.0 * k / max(w - 1, 1)
            v = -1.0 + 2.0 * j / max(h - 1, 1)
            target = torch.tensor([u, v], device=feats.device)
            dist = (uv - target).pow(2).sum(-1)
            idx = dist.argmin()
            grid[j, k] = feats[idx]
    return grid.permute(2, 0, 1).unsqueeze(0)


def dynamic_latent_sampling(uv: Tensor, feats: Tensor) -> Tensor:
    """
    Algorithm 1 stub: center-first ordering, fill sqrt(M)×sqrt(M) grid.
    Returns perspective latent [1, C, H, W].
    """
    m = uv.shape[0]
    norms = uv.norm(dim=-1)
    order = torch.argsort(norms)
    h = w = max(1, int(math.floor(math.sqrt(m))))
    c = feats.shape[-1]
    sorted_feats = feats[order]
    cap = min(h * w, sorted_feats.shape[0])
    flat = torch.zeros(h * w, c, device=feats.device)
    flat[:cap] = sorted_feats[:cap]
    return flat.view(h, w, c).permute(2, 0, 1).unsqueeze(0)


def map_view_latent(
    dirs: Tensor,
    feats: Tensor,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    fov_deg: float,
    h: int,
    w: int,
    dynamic: bool,
) -> Tensor:
    uv, mask = project_to_perspective(
        dirs,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        fov_deg=fov_deg,
    )
    uv = uv[mask]
    f = feats[mask]
    if f.numel() == 0:
        return torch.zeros(1, feats.shape[-1], h, w, device=feats.device)
    if dynamic:
        lat = dynamic_latent_sampling(uv, f)
    else:
        lat = nearest_sampling(uv, f, h, w)
    if lat.shape[-2:] != (h, w):
        lat = F.interpolate(lat, size=(h, w), mode="bilinear", align_corners=False)
    return lat
