"""Fold 360Anything ERP conditioning readiness into latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    has_pers = meta.get("perspective_ref") is not None or meta.get("has_perspective")
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    tokens = int(meta.get("token_count") or 0)
    tok_score = float(np.clip(tokens / 256.0, 0.0, 1.0)) if tokens else (0.6 if is_erp else 0.25)
    readiness = float(np.clip(0.4 * float(is_erp) + 0.35 * tok_score + (0.25 if has_pers else 0.0), 0.0, 1.0))
    out["anything360"] = {
        "arxiv_id": "2505.03296",
        "conditioning_readiness": round(readiness, 4),
        "is_erp": is_erp,
        "has_perspective_ref": bool(has_pers),
    }
    return out
