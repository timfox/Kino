"""Fold Sphere-Depth gravity-aligned calibration readiness into ERP shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    gravity = meta.get("gravity_vector") or hdr_meta.get("gravity_vector")
    has_pose = meta.get("camera_pose") is not None or meta.get("pitch_deg") is not None
    has_depth = data.get("depth_erp") is not None or meta.get("has_depth")
    if hasattr(has_depth, "numel"):
        has_depth = bool(has_depth.numel() > 0)
    grav_ok = gravity is not None or has_pose
    depth_ok = bool(has_depth)
    readiness = float(np.clip(0.35 * float(grav_ok) + 0.35 * float(depth_ok) + 0.3 * float(meta.get("erp", False)), 0.0, 1.0))
    out["sphere_depth"] = {
        "arxiv_id": "2604.23432",
        "calibration_readiness": round(readiness, 4),
        "has_gravity_prior": bool(grav_ok),
        "has_depth_map": bool(depth_ok),
    }
    return out
