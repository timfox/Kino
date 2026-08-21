"""Build TriSplat meshes from GOPEX SfM exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ltx_trainer.trisplat.cameras import TriSplatCamera, load_cameras_from_sfm, relative_first_view
from ltx_trainer.trisplat.config import TriSplatConfig
from ltx_trainer.trisplat.depth import depth_from_sparse_points, load_sparse_ply
from ltx_trainer.trisplat.geometry import pointmap_from_depth, tangent_frames_grid
from ltx_trainer.trisplat.normals import compute_view_normals
from ltx_trainer.trisplat.triangles import TrianglePrimitive, build_world_triangles, default_attributes


@dataclass
class TriSplatSceneResult:
    cameras: list[TriSplatCamera]
    triangles: list[TrianglePrimitive]
    per_view_depth: list[np.ndarray]
    export_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_cameras": len(self.cameras),
            "num_triangles": len(self.triangles),
            "export_stats": self.export_stats,
        }


def reconstruct_from_sfm(
    sfm_dir: Path,
    *,
    cfg: TriSplatConfig | None = None,
    max_views: int | None = 12,
    images_root: Path | None = None,
) -> TriSplatSceneResult:
    """Geometry-bootstrap TriSplat from sparse SfM + dense depth lift."""
    cfg = cfg or TriSplatConfig()
    sfm_dir = Path(sfm_dir)
    cameras = relative_first_view(load_cameras_from_sfm(sfm_dir, images_root))
    if max_views is not None:
        cameras = cameras[:max_views]

    ply = sfm_dir / "sparse.ply"
    points = load_sparse_ply(ply) if ply.is_file() else np.zeros((0, 3), dtype=np.float32)

    all_tris: list[TrianglePrimitive] = []
    depths: list[np.ndarray] = []

    for cam in cameras:
        import cv2  # noqa: PLC0415

        bgr = cv2.imread(str(cam.image_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if rgb.shape[0] != cam.height or rgb.shape[1] != cam.width:
            rgb = cv2.resize(rgb, (cam.width, cam.height), interpolation=cv2.INTER_AREA)

        depth = depth_from_sparse_points(cam, points)
        depths.append(depth)
        pmap_cam = pointmap_from_depth(depth, cam.K)
        attrs = default_attributes(cam.height, cam.width, seed=cam.view_id)
        attrs.color = np.clip(rgb, 0, 1)
        normals, mask = compute_view_normals(pmap_cam, rgb, step=100000, bootstrap_cfg=cfg.bootstrap)
        R_local = tangent_frames_grid(normals, pmap_cam, mask)
        tris = build_world_triangles(
            pmap_cam, attrs, cam.R_cw, cam.t_cw, cam.K, R_local, mask, cfg=cfg, rgb=rgb
        )
        all_tris.extend(tris)

    stats = {"views": len(cameras), "sparse_points": int(points.shape[0])}
    return TriSplatSceneResult(
        cameras=cameras,
        triangles=all_tris,
        per_view_depth=depths,
        export_stats=stats,
    )
