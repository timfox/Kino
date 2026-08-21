"""Fold PanoGSDet gaussian detection readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    gaussians = int(meta.get("gaussian_count") or meta.get("gs_count") or 0)
    proposals = int(meta.get("det_proposals") or 0)
    gs_score = float(np.clip(gaussians / 10000.0, 0.0, 1.0)) if gaussians else (0.45 if is_erp else 0.15)
    prop_score = float(np.clip(proposals / 100.0, 0.0, 1.0)) if proposals else 0.0
    readiness = float(np.clip(0.5 * gs_score + 0.3 * prop_score + 0.2 * float(is_erp), 0.0, 1.0))
    out["panogsdet"] = {
        "arxiv_id": "2605.14601",
        "gs_detection_readiness": round(readiness, 4),
        "gaussian_count": gaussians,
    }
    return out
