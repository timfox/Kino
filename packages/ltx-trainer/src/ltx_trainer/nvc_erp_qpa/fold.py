"""Fold NVC ERP QPA compression readiness into 360° video shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360") or meta.get("projection") == "erp")
    q_num = int(meta.get("q_num") or meta.get("lambda_bins") or 0)
    pole_q = float(meta.get("q_pole") or 0.0)
    equator_q = float(meta.get("q_equator") or 0.0)
    q_score = float(np.clip(q_num / 64.0, 0.0, 1.0)) if q_num else (0.65 if is_erp else 0.2)
    adapt_score = 0.0
    if pole_q > 0 and equator_q > 0:
        adapt_score = float(np.clip(abs(equator_q - pole_q) / max(equator_q, 1e-6), 0.0, 1.0))
    readiness = float(np.clip(0.5 * q_score + 0.3 * float(is_erp) + 0.2 * adapt_score, 0.0, 1.0))
    out["nvc_erp_qpa"] = {
        "arxiv_id": "2512.20093",
        "qpa_readiness": round(readiness, 4),
        "is_erp": is_erp,
        "q_num": q_num,
    }
    return out
