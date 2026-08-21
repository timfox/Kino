"""TriSplat inference API (trained checkpoint hook + geometry bootstrap)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ltx_trainer.trisplat.cameras import TriSplatCamera
from ltx_trainer.trisplat.config import TriSplatConfig
from ltx_trainer.trisplat.export import export_mesh_obj, export_mesh_ply
from ltx_trainer.trisplat.geometry import pointmap_from_depth, tangent_frames_grid
from ltx_trainer.trisplat.normals import compute_view_normals
from ltx_trainer.trisplat.rasterize import render_triangles
from ltx_trainer.trisplat.sfm_bridge import TriSplatSceneResult, reconstruct_from_sfm
from ltx_trainer.trisplat.triangles import PerPixelAttributes, TrianglePrimitive, build_world_triangles, default_attributes


@dataclass
class TriSplatMeshExport:
    obj_path: Path
    ply_path: Path | None
    stats: dict


def reconstruct_view_triangles(
    pointmap_cam: np.ndarray,
    rgb: np.ndarray,
    camera: TriSplatCamera,
    *,
    cfg: TriSplatConfig | None = None,
    attrs: PerPixelAttributes | None = None,
    teacher_normals: np.ndarray | None = None,
    step: int = 100000,
) -> list[TrianglePrimitive]:
    cfg = cfg or TriSplatConfig()
    attrs = attrs or default_attributes(rgb.shape[0], rgb.shape[1])
    normals, mask = compute_view_normals(
        pointmap_cam,
        rgb,
        teacher=teacher_normals,
        step=step,
        bootstrap_cfg=cfg.bootstrap,
    )
    R_local = tangent_frames_grid(normals, pointmap_cam, mask)
    return build_world_triangles(
        pointmap_cam,
        attrs,
        camera.R_cw,
        camera.t_cw,
        camera.K,
        R_local,
        mask,
        cfg=cfg,
        step=step,
        rgb=rgb,
    )


def export_simulation_mesh(
    triangles: list[TrianglePrimitive],
    output_prefix: Path,
    *,
    cfg: TriSplatConfig | None = None,
    write_ply: bool = True,
) -> TriSplatMeshExport:
    cfg = cfg or TriSplatConfig()
    output_prefix = Path(output_prefix)
    obj = output_prefix.with_suffix(".obj")
    stats = export_mesh_obj(triangles, obj, cfg=cfg.export)
    ply_path = None
    if write_ply:
        ply_path = output_prefix.with_suffix(".ply")
        stats_ply = export_mesh_ply(triangles, ply_path, cfg=cfg.export)
        stats.update(stats_ply)
    return TriSplatMeshExport(obj_path=obj, ply_path=ply_path, stats=stats)


def render_novel_view(
    triangles: list[TrianglePrimitive],
    camera: TriSplatCamera,
) -> np.ndarray:
    rgb, _, _ = render_triangles(triangles, camera)
    return rgb


# Type alias for future PI3 / official TriSplat checkpoint loader
TriSplatPredictor = Callable[[list[np.ndarray]], TriSplatSceneResult]


def reconstruct_from_images(
    image_paths: list[Path],
    output_dir: Path,
    *,
    cfg: TriSplatConfig | None = None,
    run_sfm: bool = True,
    predictor: TriSplatPredictor | None = None,
) -> TriSplatMeshExport:
    """End-to-end: images → (optional SfM) → triangle mesh."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if predictor is not None:
        images = []
        import cv2  # noqa: PLC0415

        for p in image_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise FileNotFoundError(p)
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        scene = predictor(images)
        return export_simulation_mesh(scene.triangles, output_dir / "scene_mesh", cfg=cfg)

    if run_sfm:
        frames = output_dir / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(image_paths):
            dst = frames / f"view_{i:04d}{Path(src).suffix or '.jpg'}"
            if Path(src).resolve() != dst.resolve():
                import shutil

                shutil.copy2(src, dst)
        from ltx_trainer.sfm_export import export_sfm_from_images

        sfm_out = output_dir / "sfm"
        export_sfm_from_images(frames, sfm_out, backend="auto")
        scene = reconstruct_from_sfm(sfm_out, cfg=cfg, images_root=frames)
    else:
        raise ValueError("run_sfm=False requires a TriSplatPredictor checkpoint")

    return export_simulation_mesh(scene.triangles, output_dir / "scene_mesh", cfg=cfg)
