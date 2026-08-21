"""FUSE — MCM + CVGC + ASH/RPS stateless fusion (stub)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.fuse_flow.config import FuseFlowConfig


def measurement_confidence(
    depth: torch.Tensor,
    *,
    min_depth: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Per-pixel MCM confidence (Eq. 18 simplified)."""
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    d = depth.clamp_min(min_depth)
    gx = F.pad(d[..., 1:] - d[..., :-1], (0, 1))
    gy = F.pad(d[..., 1:, :] - d[..., :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(gx * gx + gy * gy) / d
    k = 3
    local_mean = F.avg_pool2d(d, k, stride=1, padding=k // 2)
    local_var = F.avg_pool2d((d - local_mean).pow(2), k, stride=1, padding=k // 2) / (d * d)
    conf = torch.exp(-alpha * grad) * torch.exp(-beta * local_var)
    return conf.squeeze(0)


def harmonic_fusion_weight(
    mcm: torch.Tensor,
    cvgc: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Eq. (22): dual-gate harmonic mean of single-view and cross-view weights."""
    return (2.0 * mcm * cvgc) / (mcm + cvgc + eps)


def cross_view_geometric_consistency(
    depth_a: torch.Tensor,
    depth_b: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CVGC proxy: agreement of normalized depth gradients between neighboring views."""
    if depth_a.dim() == 2:
        depth_a = depth_a.unsqueeze(0)
    if depth_b.dim() == 2:
        depth_b = depth_b.unsqueeze(0)
    da = depth_a / depth_a.clamp_min(eps).amax(dim=(-2, -1), keepdim=True)
    db = depth_b / depth_b.clamp_min(eps).amax(dim=(-2, -1), keepdim=True)
    gx_a = F.pad(da[..., 1:] - da[..., :-1], (0, 1))
    gx_b = F.pad(db[..., 1:] - db[..., :-1], (0, 1))
    gy_a = F.pad(da[..., 1:, :] - da[..., :-1, :], (0, 0, 0, 1))
    gy_b = F.pad(db[..., 1:, :] - db[..., :-1, :], (0, 0, 0, 1))
    grad_a = torch.sqrt(gx_a * gx_a + gy_a * gy_a)
    grad_b = torch.sqrt(gx_b * gx_b + gy_b * gy_b)
    agree = torch.exp(-(grad_a - grad_b).abs().mean(dim=(-2, -1), keepdim=True))
    return agree.squeeze(0)


def spatial_hash_index(
    points: torch.Tensor,
    *,
    coarse_cell: float = 0.25,
    fine_base: float = 0.05,
) -> torch.Tensor:
    """Two-level hash index (coarse cell, fine cell) for 3D points."""
    coarse = torch.floor(points / coarse_cell).to(torch.long)
    local = points - coarse.float() * coarse_cell
    fine = torch.floor(local / fine_base).to(torch.long)
    return torch.cat([coarse, fine], dim=-1)


def fuse_views_stub(
    depths: list[torch.Tensor],
    *,
    cfg: FuseFlowConfig | None = None,
) -> dict[str, Any]:
    """Back-project valid pixels, score confidence, hash cells (CPU demo)."""
    cfg = cfg or FuseFlowConfig()
    confs = [measurement_confidence(d) for d in depths]
    n_valid = sum(int((c > 0.05).sum()) for c in confs)
    pts = []
    ref = depths[0]
    for i, (d, c) in enumerate(zip(depths, confs, strict=True)):
        mask = c > 0.1
        yy, xx = torch.where(mask)
        if yy.numel() == 0:
            continue
        z = d[yy, xx]
        x = (xx.float() / d.shape[-1] - 0.5) * z
        y = (yy.float() / d.shape[-2] - 0.5) * z
        p = torch.stack([x, y, z], dim=-1)
        cvgc_scalar = cross_view_geometric_consistency(ref, d).mean() if i else torch.tensor(1.0)
        cvgc = torch.full_like(c[mask], float(cvgc_scalar.item()))
        w = harmonic_fusion_weight(c[mask], cvgc)
        pts.append((p, w))
    if not pts:
        return {"n_points": 0, "n_cells": 0}
    all_p = torch.cat([p for p, _ in pts], dim=0)
    cells = spatial_hash_index(all_p)
    unique_cells = torch.unique(cells, dim=0)
    return {
        "n_views": len(depths),
        "n_valid_pixels": n_valid,
        "n_points": int(all_p.shape[0]),
        "n_cells": int(unique_cells.shape[0]),
        "mean_confidence": float(torch.cat([w for _, w in pts]).mean()),
    }
