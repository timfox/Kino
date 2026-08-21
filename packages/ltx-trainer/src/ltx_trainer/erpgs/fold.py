"""Fold ERP-GS gaussian splatting readiness into ERP shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    iters = int(meta.get("gs_iterations") or meta.get("iterations") or 0)
    views = int(meta.get("nvs_views") or meta.get("num_views") or 0)
    iter_score = float(np.clip(iters / 30000.0, 0.0, 1.0)) if iters else (0.5 if is_erp else 0.15)
    view_score = float(np.clip(views / 12.0, 0.0, 1.0)) if views else 0.3
    readiness = float(np.clip(0.4 * iter_score + 0.35 * view_score + 0.25 * float(is_erp), 0.0, 1.0))
    out["erpgs"] = {
        "arxiv_id": "2411.07135",
        "splat_readiness": round(readiness, 4),
        "is_erp": is_erp,
        "iterations": iters,
    }
    return out
