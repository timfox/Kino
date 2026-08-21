"""Fold Gimbal360 canonical ERP diffusion readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    horizon_ok = bool(meta.get("horizon_aligned") or meta.get("gravity_aligned"))
    has_flow = meta.get("optical_flow") is not None or meta.get("dal_flow")
    flow_bonus = 0.2 if bool(has_flow) else 0.0
    readiness = float(np.clip(0.4 * float(is_erp) + 0.35 * float(horizon_ok) + flow_bonus + 0.05, 0.0, 1.0))
    out["gimbal360"] = {
        "arxiv_id": "2603.23179",
        "canonical_erp_readiness": round(readiness, 4),
        "horizon_aligned": horizon_ok,
        "is_erp": is_erp,
    }
    return out
