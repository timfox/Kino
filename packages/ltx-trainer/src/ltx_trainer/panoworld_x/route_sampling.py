"""PanoExplorer data pipeline stages (Sec. 3.2) — planning stubs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def sample_route_candidate(
    num_points: int = 32,
    *,
    seed: int = 0,
) -> Tensor:
    """Random walkable polyline stub [N, 3] (x, y, z)."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.cumsum(torch.randn(num_points, 3, generator=g) * 0.5, dim=0)
    return pts


def path_length_meters(route_xyz: Tensor) -> float:
    if route_xyz.shape[0] < 2:
        return 0.0
    return float(torch.linalg.norm(route_xyz[1:] - route_xyz[:-1], dim=1).sum())


def filter_by_min_length(route_xyz: Tensor, min_meters: float = 18.0) -> bool:
    return path_length_meters(route_xyz) >= min_meters


def collision_free_stub(route_xyz: Tensor) -> bool:
    """Bounding-box proxy — always True for smoke unless z dips below floor."""
    return bool((route_xyz[:, 2] > -0.5).all())


def normalize_frame_stride(route_xyz: Tensor, stride_m: float = 0.10) -> Tensor:
    """Resample polyline to ~fixed 10 cm steps (spatial normalization stub)."""
    if route_xyz.shape[0] < 2:
        return route_xyz
    seg = torch.linalg.norm(route_xyz[1:] - route_xyz[:-1], dim=1)
    total = float(seg.sum())
    if total < 1e-6:
        return route_xyz
    n = max(2, int(total / stride_m) + 1)
    t = torch.linspace(0, 1, n)
    idx = t * (route_xyz.shape[0] - 1)
    i0 = idx.long().clamp(max=route_xyz.shape[0] - 2)
    frac = (idx - i0.float()).unsqueeze(-1)
    return route_xyz[i0] * (1 - frac) + route_xyz[i0 + 1] * frac


def pipeline_report(min_meters: float = 18.0) -> dict[str, Any]:
    route = sample_route_candidate()
    route = normalize_frame_stride(route)
    return {
        "length_m": path_length_meters(route),
        "passes_min_length": filter_by_min_length(route, min_meters),
        "collision_free": collision_free_stub(route),
        "num_points": int(route.shape[0]),
    }
