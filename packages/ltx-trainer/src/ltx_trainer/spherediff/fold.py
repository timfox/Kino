"""Fold SphereDiff spherical-latent readiness into video / ERP latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    is_360 = bool(meta.get("is_360") or meta.get("erp") or hdr_meta.get("projection") == "erp")
    num_views = int(meta.get("num_views") or meta.get("sphere_views") or 0)
    latent_density = float(meta.get("latent_density") or 0.0)
    view_score = float(np.clip(num_views / 8.0, 0.0, 1.0)) if num_views else (0.65 if is_360 else 0.2)
    density_score = float(np.clip(latent_density / 2600.0, 0.0, 1.0)) if latent_density > 0 else view_score
    readiness = float(np.clip(0.5 * view_score + 0.5 * density_score, 0.0, 1.0))
    out["spherediff"] = {
        "arxiv_id": "2504.14396",
        "spherical_readiness": round(readiness, 4),
        "is_360": is_360,
        "num_views": num_views,
    }
    return out
