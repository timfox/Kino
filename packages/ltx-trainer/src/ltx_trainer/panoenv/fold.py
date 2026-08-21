"""Fold PanoEnv spatial VQA readiness into ERP shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    qa_count = int(meta.get("qa_count") or len(meta.get("questions") or []) or 0)
    has_3d = bool(meta.get("has_3d_gt") or meta.get("tartanair"))
    qa_score = float(np.clip(qa_count / 4.0, 0.0, 1.0)) if qa_count else (0.5 if is_erp else 0.15)
    readiness = float(np.clip(0.4 * qa_score + 0.35 * float(is_erp) + (0.25 if has_3d else 0.0), 0.0, 1.0))
    out["panoenv"] = {
        "arxiv_id": "2602.21992",
        "spatial_vqa_readiness": round(readiness, 4),
        "qa_count": qa_count,
        "has_3d_gt": has_3d,
    }
    return out
