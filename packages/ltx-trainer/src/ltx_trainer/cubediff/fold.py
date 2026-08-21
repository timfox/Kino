"""Fold CubeDiff cubemap panorama readiness into video / ERP latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360") or hdr_meta.get("projection") == "erp")
    aspect = float(meta.get("aspect_ratio") or hdr_meta.get("aspect_ratio") or 0.0)
    erp_ok = aspect >= 1.8 if aspect > 0 else is_erp
    faces_ready = bool(meta.get("cubemap_faces") or data.get("cubemap_faces"))
    readiness = float(np.clip((0.55 if erp_ok else 0.2) + (0.45 if faces_ready else 0.15), 0.0, 1.0))
    out["cubediff"] = {
        "arxiv_id": "2501.17162",
        "panorama_readiness": round(readiness, 4),
        "is_erp": is_erp,
        "aspect_ratio": aspect,
        "num_faces": 6 if faces_ready else 0,
    }
    return out
