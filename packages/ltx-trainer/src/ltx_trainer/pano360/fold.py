"""Fold pano360 equirect photogrammetry readiness into video shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pano360.geometry import is_equirectangular_size


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    w = int(meta.get("erp_width") or meta.get("width") or 0)
    h = int(meta.get("erp_height") or meta.get("height") or 0)
    eq = is_equirectangular_size(w, h) if w and h else bool(meta.get("erp") or meta.get("is_360"))
    views = int(meta.get("views_per_frame") or meta.get("pano_views") or 0)
    view_score = float(np.clip(views / 8.0, 0.0, 1.0)) if views else (0.65 if eq else 0.2)
    readiness = float(np.clip(0.55 * float(eq) + 0.45 * view_score, 0.0, 1.0))
    out["pano360"] = {
        "photogrammetry_readiness": round(readiness, 4),
        "is_equirectangular": eq,
        "erp_width": w,
        "erp_height": h,
        "views_per_frame": views,
    }
    return out
