"""Fold SphereFusion panorama depth readiness into ERP latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    h = int(meta.get("erp_height") or hdr_meta.get("erp_height") or 0)
    w = int(meta.get("erp_width") or hdr_meta.get("erp_width") or 0)
    aspect = w / max(h, 1) if h and w else float(meta.get("aspect_ratio") or 0.0)
    erp_ok = aspect >= 1.8 or bool(meta.get("erp") or meta.get("is_360"))
    has_depth = data.get("depth_erp") is not None or meta.get("has_panorama_depth")
    depth_flag = bool(has_depth) if not hasattr(has_depth, "numel") else bool(np.asarray(has_depth).size > 0)
    readiness = float(np.clip((0.55 if erp_ok else 0.15) + (0.45 if depth_flag else 0.1), 0.0, 1.0))
    out["spherefusion"] = {
        "arxiv_id": "2502.05859",
        "depth_readiness": round(readiness, 4),
        "erp_aspect": round(aspect, 3) if aspect else 0.0,
        "has_depth_prior": depth_flag,
    }
    return out
