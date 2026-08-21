"""Depth maps from sparse SfM points (bootstrap without trained TriSplat net)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ltx_trainer.trisplat.cameras import TriSplatCamera


def depth_from_sparse_points(
    camera: TriSplatCamera,
    points_world: np.ndarray,
    *,
    default_depth: float = 3.0,
) -> np.ndarray:
    """Fill depth map by splatting sparse 3D points into the view."""
    h, w = camera.height, camera.width
    depth = np.full((h, w), default_depth, dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    if points_world.size == 0:
        return depth

    pts_cam = camera.world_to_camera(points_world)
    z = pts_cam[:, 2]
    valid = z > 0.1
    pts_cam = pts_cam[valid]
    z = z[valid]
    u, v = camera.project(pts_cam)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    for i in range(len(z)):
        x, y = int(ui[i]), int(vi[i])
        if 0 <= x < w and 0 <= y < h:
            if weight[y, x] == 0 or z[i] < depth[y, x]:
                depth[y, x] = float(z[i])
                weight[y, x] = 1.0

    try:
        import cv2  # noqa: PLC0415

        mask = (weight > 0).astype(np.uint8)
        depth_inpaint = cv2.inpaint(
            depth.astype(np.float32),
            (1 - mask).astype(np.uint8),
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA,
        )
        depth = np.where(mask > 0, depth, depth_inpaint)
    except ImportError:
        pass
    return depth.astype(np.float32)


def load_sparse_ply(path: Path) -> np.ndarray:
    path = Path(path)
    verts: list[list[float]] = []
    in_header = True
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if in_header:
            if line.strip() == "end_header":
                in_header = False
            continue
        parts = line.split()
        if len(parts) >= 3:
            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(verts, dtype=np.float32) if verts else np.zeros((0, 3), dtype=np.float32)
