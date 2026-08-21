"""Software triangle splat rasterizer (TriSplat / Triangle Splatting style)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.trisplat.cameras import TriSplatCamera
from ltx_trainer.trisplat.triangles import TrianglePrimitive


def _project_triangle(
    verts_world: np.ndarray,
    cam: TriSplatCamera,
) -> tuple[np.ndarray, float]:
    v_cam = cam.world_to_camera(verts_world)
    z_mean = float(v_cam[:, 2].mean())
    u, v = cam.project(v_cam)
    return np.stack([u, v], axis=1), z_mean


def _edge_fn(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Signed edge function for triangle edge ``a→b`` at point ``p``."""
    return (p[0] - b[0]) * (a[1] - b[1]) - (p[1] - b[1]) * (a[0] - b[0])


def render_triangles(
    triangles: list[TrianglePrimitive],
    camera: TriSplatCamera,
    *,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render RGB ``(H,W,3)``, depth ``(H,W)``, normals ``(H,W,3)``."""
    H, W = camera.height, camera.width
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    normal_acc = np.zeros((H, W, 3), dtype=np.float32)
    alpha_acc = np.zeros((H, W), dtype=np.float32)

    sorted_tris: list[tuple[float, TrianglePrimitive, np.ndarray]] = []
    for tri in triangles:
        uv, z = _project_triangle(tri.vertices, camera)
        if z < 1e-3:
            continue
        if np.all(uv[:, 0] < -1) or np.all(uv[:, 0] > W + 1) or np.all(uv[:, 1] < -1) or np.all(uv[:, 1] > H + 1):
            continue
        sorted_tris.append((z, tri, uv))
    sorted_tris.sort(key=lambda x: x[0], reverse=True)

    for _, tri, uv in sorted_tris:
        min_u, min_v = uv.min(axis=0)
        max_u, max_v = uv.max(axis=0)
        pad = int(max(3, tri.blur * 8))
        x0 = max(0, int(min_u) - pad)
        x1 = min(W, int(max_u) + pad + 1)
        y0 = max(0, int(min_v) - pad)
        y1 = min(H, int(max_v) + pad + 1)
        area = _edge_fn(uv[1], uv[2], uv[0])
        if abs(area) < 1e-8:
            continue
        face_n = np.cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0])
        fn = np.linalg.norm(face_n)
        if fn < 1e-8:
            continue
        face_n = face_n / fn

        for y in range(y0, y1):
            for x in range(x0, x1):
                p = np.array([float(x), float(y)], dtype=np.float64)
                w0 = _edge_fn(uv[1], uv[2], p) / area
                w1 = _edge_fn(uv[2], uv[0], p) / area
                w2 = _edge_fn(uv[0], uv[1], p) / area
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                dist_edge = min(w0, w1, w2) * max(abs(area), 1e-6) ** 0.5
                soft = np.exp(-max(0.0, dist_edge) / max(tri.blur, 1e-3))
                alpha = tri.opacity * soft
                alpha = min(0.99, max(0.0, alpha))
                a_prev = alpha_acc[y, x]
                weight = alpha * (1.0 - a_prev)
                rgb[y, x] += tri.color * weight
                normal_acc[y, x] += face_n.astype(np.float32) * weight
                z_pix = float(np.dot(tri.vertices.mean(axis=0) - camera.t_cw, face_n))
                if weight > 0.01:
                    depth[y, x] = min(depth[y, x], z_pix)
                alpha_acc[y, x] = a_prev + weight

    bg = np.array(bg_color, dtype=np.float32)
    rgb = rgb + bg * (1.0 - alpha_acc[..., None])
    norm_len = np.linalg.norm(normal_acc, axis=-1, keepdims=True)
    normals = np.zeros_like(normal_acc)
    ok = norm_len[..., 0] > 1e-6
    normals[ok] = normal_acc[ok] / norm_len[ok]
    depth[~np.isfinite(depth)] = 0
    return rgb, depth, normals
