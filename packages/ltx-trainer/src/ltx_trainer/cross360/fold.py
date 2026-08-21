"""Fold Cross360 depth fusion readiness into ERP shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    tp_n = int(meta.get("tp_N") or meta.get("transformer_points") or 0)
    has_depth = meta.get("depth_erp") is not None or meta.get("has_depth")
    tp_score = float(np.clip(tp_n / 1024.0, 0.0, 1.0)) if tp_n else (0.55 if is_erp else 0.2)
    depth_bonus = 0.2 if bool(has_depth) else 0.0
    readiness = float(np.clip(0.45 * tp_score + 0.35 * float(is_erp) + depth_bonus, 0.0, 1.0))
    out["cross360"] = {
        "arxiv_id": "2503.08926",
        "depth_fusion_readiness": round(readiness, 4),
        "is_erp": is_erp,
        "tp_N": tp_n,
    }
    return out
