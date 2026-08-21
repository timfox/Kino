"""Support-plane grid partitioning + camera ranking (Sec. 3.2, Supp. 6)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def fit_dominant_plane_ransac(points: Tensor, *, threshold: float = 0.05, iters: int = 32) -> tuple[Tensor, Tensor]:
    """RANSAC plane n·x + d = 0; returns unit normal n and offset d."""
    if points.shape[0] < 3:
        n = torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype, device=points.device)
        return n, torch.tensor(0.0, dtype=points.dtype, device=points.device)
    best_inl = 0
    best_n = torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype, device=points.device)
    best_d = torch.tensor(0.0, dtype=points.dtype, device=points.device)
    g = torch.Generator(device=points.device)
    for _ in range(iters):
        idx = torch.randperm(points.shape[0], generator=g)[:3]
        p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]
        v1, v2 = p1 - p0, p2 - p0
        n = torch.linalg.cross(v1, v2)
        if n.norm() < 1e-8:
            continue
        n = n / n.norm()
        d = -n.dot(p0)
        dist = (points @ n + d).abs()
        inl = int((dist < threshold).sum())
        if inl > best_inl:
            best_inl, best_n, best_d = inl, n, d
    return best_n, best_d


def project_to_plane(points: Tensor, origin: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Map 3D points to (u, v) planar coordinates."""
    rel = points - origin.unsqueeze(0)
    return torch.stack([rel @ u, rel @ v], dim=-1)


def grid_partitions(
    planar_uv: Tensor,
    *,
    rows: int,
    cols: int,
    inflate_u: float = 0.1,
    inflate_v: float = 0.1,
) -> list[Tensor]:
    """Return boolean masks (N,) per cell with inflated windows."""
    uv_min = planar_uv.min(dim=0).values
    uv_max = planar_uv.max(dim=0).values
    span_u = (uv_max[0] - uv_min[0]).clamp(min=1e-4)
    span_v = (uv_max[1] - uv_min[1]).clamp(min=1e-4)
    masks: list[Tensor] = []
    for r in range(rows):
        for c in range(cols):
            uc0 = uv_min[0] + c / cols * span_u
            uc1 = uv_min[0] + (c + 1) / cols * span_u
            vc0 = uv_min[1] + r / rows * span_v
            vc1 = uv_min[1] + (r + 1) / rows * span_v
            du, dv = (uc1 - uc0) * inflate_u, (vc1 - vc0) * inflate_v
            um = (uc0 + uc1) / 2
            vm = (vc0 + vc1) / 2
            u0, u1 = um - (uc1 - uc0 + du) / 2, um + (uc1 - uc0 + du) / 2
            v0, v1 = vm - (vc1 - vc0 + dv) / 2, vm + (vc1 - vc0 + dv) / 2
            mask = (planar_uv[:, 0] >= u0) & (planar_uv[:, 0] <= u1) & (planar_uv[:, 1] >= v0) & (
                planar_uv[:, 1] <= v1
            )
            masks.append(mask)
    return masks


def rank_cameras_for_partition(
    point_indices: Tensor,
    observations: dict[int, list[int]],
    *,
    top_m: int = 16,
    top_k_pairs: int = 3,
) -> list[int]:
    """Toy camera ranking via co-visibility pair votes (Eq. 29–31)."""
    if point_indices.numel() == 0:
        return []
    cams: set[int] = set()
    for pi in point_indices.tolist():
        cams.update(observations.get(int(pi), []))
    if not cams:
        return []
    cam_list = sorted(cams)
    scores = {c: 0.0 for c in cam_list}
    for pi in point_indices.tolist():
        obs = [c for c in observations.get(int(pi), []) if c in scores]
        pairs: list[tuple[float, int, int]] = []
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                a, b = obs[i], obs[j]
                pairs.append((1.0, a, b))
        pairs.sort(reverse=True)
        for w, a, b in pairs[:top_k_pairs]:
            scores[a] += w / 2
            scores[b] += w / 2
    ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return ranked[:top_m]
