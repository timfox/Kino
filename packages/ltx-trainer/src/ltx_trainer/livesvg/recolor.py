"""Sphere-packing path recolorization (Sec. 3.2.2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def farthest_point_rgb_palette(
    num_paths: int,
    *,
    seed: int = 0,
    candidate_pool: int = 512,
) -> Tensor:
    """High-separation RGB palette in [0, 1]^3 (Packomania-style greedy packing)."""
    if num_paths < 1:
        raise ValueError("num_paths must be positive")
    gen = torch.Generator().manual_seed(seed)
    chosen = torch.rand(3, generator=gen).unsqueeze(0)
    while chosen.shape[0] < num_paths:
        candidates = torch.rand(candidate_pool, 3, generator=gen)
        # min distance to any chosen center
        dist = torch.cdist(candidates, chosen).amin(dim=1)
        idx = int(dist.argmax().item())
        chosen = torch.cat([chosen, candidates[idx : idx + 1]], dim=0)
    return chosen[:num_paths]


def min_pairwise_rgb_distance(palette: Tensor) -> float:
    """Minimum Euclidean distance between palette rows."""
    if palette.shape[0] < 2:
        return float("inf")
    d = torch.cdist(palette, palette)
    d = d + torch.eye(d.shape[0], device=d.device) * 1e6
    return float(d.min().item())


def hsv_distinct_fallback(num_paths: int) -> Tensor:
    """HSV distinct-color fallback when packing table size is unavailable."""
    hues = torch.linspace(0.0, 1.0 - 1.0 / max(num_paths, 1), num_paths)
    colors = []
    for h in hues:
        rgb = _hsv_to_rgb(float(h), 0.85, 0.95)
        colors.append(rgb)
    return torch.tensor(colors, dtype=torch.float32)


def assign_path_palette(
    num_paths: int,
    *,
    seed: int = 0,
    use_hsv_fallback: bool = False,
) -> Tensor:
    if use_hsv_fallback:
        return hsv_distinct_fallback(num_paths)
    return farthest_point_rgb_palette(num_paths, seed=seed)


def _hsv_to_rgb(h: float, s: float, v: float) -> list[float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return [r, g, b]


def recolorization_report(num_paths: int, *, seed: int = 0) -> dict[str, float]:
    palette = assign_path_palette(num_paths, seed=seed)
    return {
        "num_paths": float(num_paths),
        "min_rgb_separation": min_pairwise_rgb_distance(palette),
        "packing_radius_lower_bound": min_pairwise_rgb_distance(palette) / 2.0,
    }
