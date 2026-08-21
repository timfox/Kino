"""Fold Pano-Affordance map readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    has_map = bool(data.get("affordance_map") is not None or meta.get("has_affordance"))
    classes = int(meta.get("affordance_classes") or 0)
    cls_score = float(np.clip(classes / 18.0, 0.0, 1.0)) if classes else (0.4 if has_map else 0.0)
    readiness = float(np.clip(0.45 * float(has_map) + 0.35 * cls_score + 0.2 * float(is_erp), 0.0, 1.0))
    out["pano_affordance"] = {
        "arxiv_id": "2603.09760",
        "affordance_readiness": round(readiness, 4),
        "has_affordance_map": bool(has_map),
    }
    return out
