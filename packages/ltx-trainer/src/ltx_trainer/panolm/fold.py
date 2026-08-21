"""Fold PanoLM panoramic language model readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    qa_total = int(meta.get("panovqa_total") or meta.get("qa_count") or 0)
    has_caption = bool(meta.get("caption") or meta.get("panorama_caption"))
    qa_score = float(np.clip(qa_total / 1000.0, 0.0, 1.0)) if qa_total else (0.35 if has_caption else 0.1)
    readiness = float(np.clip(0.45 * qa_score + 0.35 * float(is_erp) + (0.2 if has_caption else 0.0), 0.0, 1.0))
    out["panolm"] = {
        "arxiv_id": "2603.09573",
        "panovqa_readiness": round(readiness, 4),
        "panovqa_total": qa_total,
    }
    return out
