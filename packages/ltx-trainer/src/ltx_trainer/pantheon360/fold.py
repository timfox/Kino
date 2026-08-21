"""Fold Pantheon360 3D-cache fusion readiness into 360° video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_360 = bool(meta.get("is_360") or meta.get("erp"))
    num_frames = int(meta.get("num_frames") or 0)
    yaw_crops = int(meta.get("clip_yaw_crops") or meta.get("yaw_crops") or 0)
    has_cache = meta.get("pi3_cache") is not None or meta.get("v_geo_frames") is not None
    frame_score = float(np.clip(num_frames / 25.0, 0.0, 1.0)) if num_frames else (0.6 if is_360 else 0.2)
    crop_score = float(np.clip(yaw_crops / 8.0, 0.0, 1.0)) if yaw_crops else (0.5 if is_360 else 0.15)
    cache_bonus = 0.15 if has_cache else 0.0
    readiness = float(np.clip(0.45 * frame_score + 0.4 * crop_score + cache_bonus, 0.0, 1.0))
    out["pantheon360"] = {
        "arxiv_id": "2605.25449",
        "cache_fusion_readiness": round(readiness, 4),
        "num_frames": num_frames,
        "yaw_crops": yaw_crops,
        "is_360": is_360,
    }
    return out
