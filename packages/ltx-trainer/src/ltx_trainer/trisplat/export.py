"""Direct mesh export from triangle primitives (TriSplat Sec. 3.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ltx_trainer.trisplat.config import MeshExportConfig
from ltx_trainer.trisplat.triangles import TrianglePrimitive


def _quantize_key(v: np.ndarray, precision: float, normal: np.ndarray) -> tuple[int, int, int, int, int, int]:
    q = np.round(v / precision).astype(np.int64)
    n = np.sign(normal).astype(np.int8)
    return int(q[0]), int(q[1]), int(q[2]), int(n[0]), int(n[1]), int(n[2])


def export_mesh_obj(
    triangles: list[TrianglePrimitive],
    path: Path,
    *,
    cfg: MeshExportConfig | None = None,
    face_normals: np.ndarray | None = None,
) -> dict[str, int]:
    """Write Wavefront OBJ; merge vertices with quantized hashing."""
    cfg = cfg or MeshExportConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vert_map: dict[tuple, int] = {}
    verts: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    kept = 0

    for tri in triangles:
        if tri.opacity < cfg.opacity_threshold:
            continue
        fn = np.cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0])
        fn_len = np.linalg.norm(fn)
        if fn_len < 1e-8:
            continue
        fn = fn / fn_len
        idx: list[int] = []
        for k in range(3):
            key = _quantize_key(tri.vertices[k], cfg.vertex_quantize, fn)
            if key not in vert_map:
                vert_map[key] = len(verts) + 1
                verts.append(tri.vertices[k])
                colors.append(tri.color)
            idx.append(vert_map[key])
        if len(idx) == 3 and idx[0] != idx[1] != idx[2]:
            faces.append((idx[0], idx[1], idx[2]))
            kept += 1

    with path.open("w", encoding="utf-8") as f:
        f.write("# TriSplat export\n")
        for v, c in zip(verts, colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")

    return {"triangles_kept": kept, "vertices": len(verts), "faces": len(faces)}


def export_mesh_ply(triangles: list[TrianglePrimitive], path: Path, *, cfg: MeshExportConfig | None = None) -> dict[str, int]:
    cfg = cfg or MeshExportConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts_list: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    colors: list[np.ndarray] = []
    vert_map: dict[tuple, int] = {}

    for tri in triangles:
        if tri.opacity < cfg.opacity_threshold:
            continue
        fn = np.cross(tri.vertices[1] - tri.vertices[0], tri.vertices[2] - tri.vertices[0])
        fn_len = np.linalg.norm(fn)
        if fn_len < 1e-8:
            continue
        fn = fn / fn_len
        idx = []
        for k in range(3):
            key = _quantize_key(tri.vertices[k], cfg.vertex_quantize, fn)
            if key not in vert_map:
                vert_map[key] = len(verts_list)
                verts_list.append(tri.vertices[k])
                colors.append((tri.color * 255).astype(np.uint8))
            idx.append(vert_map[key])
        if len(set(idx)) == 3:
            faces.append((idx[0], idx[1], idx[2]))

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts_list)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(verts_list, colors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")

    return {"vertices": len(verts_list), "faces": len(faces)}
