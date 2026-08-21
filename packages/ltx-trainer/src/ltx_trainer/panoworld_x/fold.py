"""Fold PanoWorld-X explorable video readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    traj_m = float(meta.get("trajectory_meters") or meta.get("path_length_m") or 0.0)
    frames = int(meta.get("num_frames") or 0)
    traj_score = float(np.clip(traj_m / 18.0, 0.0, 1.0)) if traj_m > 0 else (0.5 if is_erp else 0.15)
    frame_score = float(np.clip(frames / 49.0, 0.0, 1.0)) if frames else traj_score
    readiness = float(np.clip(0.45 * frame_score + 0.35 * traj_score + 0.2 * float(is_erp), 0.0, 1.0))
    out["panoworld_x"] = {
        "arxiv_id": "2509.24997",
        "explorable_world_readiness": round(readiness, 4),
        "trajectory_meters": traj_m,
        "num_frames": frames,
    }
    return out
