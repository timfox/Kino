"""3D Cache reconstruction and geometric scaffold rendering (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PointCloudCache:
    """Explicit 3D point cloud cache (positions + optional colors)."""

    points: Tensor  # (N, 3)
    colors: Tensor | None = None  # (N, 3)


def build_cache_from_erp(
    frame: Tensor,
    *,
    num_views: int = 16,
) -> PointCloudCache:
    """
    Stub: convert a single ERP frame into a toy point cloud.

    Full pipeline crops ERP → perspective views → PI3/VGGT; here we subsample pixels.
    """
    if frame.dim() == 3:
        h, w, _ = frame.shape
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        depth = 1.0 + 0.2 * frame[..., 0]
        pts = torch.stack([xx * depth, yy * depth, depth], dim=-1).reshape(-1, 3)
        step = max(1, pts.shape[0] // (num_views * 1024))
        pts = pts[::step]
        colors = frame.reshape(-1, 3)[::step]
        return PointCloudCache(points=pts, colors=colors)
    raise ValueError("frame must be H×W×3")


def render_trajectory_erp(
    cache: PointCloudCache,
    trajectory: Tensor,
    *,
    t_frames: int,
    height: int = 512,
    width: int = 1024,
) -> Tensor:
    """
    Render geometry-only video ``V_geo`` ∈ R^{T×3×H×W} along ``C_target``.

    ``trajectory`` shape (T, 4, 4) or (T, 7) stub — we use simple orbit for demo.
    """
    frames: list[Tensor] = []
    pts = cache.points
    for t in range(t_frames):
        angle = 2.0 * 3.14159 * t / max(t_frames, 1)
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rot = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        warped = pts @ rot.T
        z = warped[:, 2:3].clamp(min=0.1)
        u = ((warped[:, 0:1] / z + 1) * 0.5 * (width - 1)).long().clamp(0, width - 1)
        v = ((warped[:, 1:2] / z + 1) * 0.5 * (height - 1)).long().clamp(0, height - 1)
        img = torch.zeros(height, width, 3)
        if cache.colors is not None:
            img[v.squeeze(-1), u.squeeze(-1)] = cache.colors[: v.shape[0]]
        else:
            img[v.squeeze(-1), u.squeeze(-1)] = torch.tensor([0.6, 0.6, 0.6])
        frames.append(img)
    return torch.stack(frames, dim=0)
