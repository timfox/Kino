"""Toy RadiusFPS vs standard FPS demo on synthetic point clouds."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.radiusfps.benchmarks import summary_anchors
from ltx_trainer.radiusfps.config import RadiusFpsConfig
from ltx_trainer.radiusfps.fps import argmax_dist, l2, standard_fps
from ltx_trainer.radiusfps.voxel import build_active_voxels, radiusfps_update_iteration, sync_dist_v


def _dragon_like_points(n: int = 256) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    for i in range(n):
        t = 2 * math.pi * i / n
        r = 0.5 + 0.3 * math.sin(3 * t)
        x = r * math.cos(t)
        y = r * math.sin(t)
        z = 0.2 * math.sin(5 * t)
        pts.append((x, y, z))
    return pts


def radiusfps_sample(
    points: list[tuple[float, float, float]],
    m: int,
    *,
    nvox: int = 16,
    seed_index: int = 0,
) -> tuple[list[int], dict[str, int]]:
    n = len(points)
    m = min(m, n)
    voxels, _, _, _ = build_active_voxels(points, nvox=nvox)
    dist_p = [float("inf")] * n
    selected: list[int] = []
    start = seed_index % n
    selected.append(start)
    dist_p[start] = 0.0
    stats = {"pruned_voxels": 0, "visited_updates": 0}
    for i in range(n):
        if i != start:
            dist_p[i] = l2(points[i], points[start])
    sync_dist_v(voxels, dist_p)
    for _ in range(1, m):
        nxt = argmax_dist(dist_p)
        selected.append(nxt)
        dist_p, pruned, visited = radiusfps_update_iteration(points, voxels, dist_p, nxt)
        stats["pruned_voxels"] += pruned
        stats["visited_updates"] += visited
    return selected, stats


def run_demo(cfg: RadiusFpsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RadiusFpsConfig()
    points = _dragon_like_points(256)
    m = cfg.num_samples
    std = standard_fps(points, m, seed_index=cfg.seed_index)
    rad, stats = radiusfps_sample(points, m, nvox=cfg.nvox, seed_index=cfg.seed_index)
    summary = summary_anchors()
    return {
        "config": {
            "num_samples": m,
            "nvox": cfg.nvox,
            "num_points": len(points),
        },
        "standard_fps_indices_head": std[:8],
        "radiusfps_indices_head": rad[:8],
        "exact_match": std == rad,
        "pruning_stats": stats,
        "summary": summary,
        "beats_gpu_fps_e2e": summary["kitti_e2e_speedup_radiusfps_g"] >= 2.0,
    }
