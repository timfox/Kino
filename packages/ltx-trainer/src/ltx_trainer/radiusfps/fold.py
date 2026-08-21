"""AV-fold sidecar: point cloud / LiDAR / FPS sampling hints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.radiusfps.constants import PAPER_ARXIV


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("radiusfps", "farthest point", "point cloud sampling", "lidar fps")):
        regime = "radiusfps"
    elif any(w in caption for w in ("pointnet", "point cloud", "lidar", "slam", "scannet", "semantickitti")):
        regime = "point_cloud_perception"
    elif any(w in caption for w in ("voxel", "3d scan", "depth cloud")):
        regime = "voxel_point_cloud"
    else:
        regime = "generic_3d"
    out["radiusfps"] = {
        "arxiv_id": PAPER_ARXIV,
        "regime_hint": regime,
        "sampler": "spherical_voxel_pruning_fps",
    }
    return out
