"""Fold Dense360 omnidirectional VLM readiness into ERP latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    captions = meta.get("entity_captions") or meta.get("captions")
    refs = meta.get("referring_expressions") or meta.get("ref_expressions")
    n_cap = len(captions) if captions is not None and hasattr(captions, "__len__") else int(meta.get("caption_count") or 0)
    n_ref = len(refs) if refs is not None and hasattr(refs, "__len__") else int(meta.get("ref_count") or 0)
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    cap_score = float(np.clip(n_cap / 5.0, 0.0, 1.0)) if n_cap else (0.4 if is_erp else 0.1)
    ref_score = float(np.clip(n_ref / 3.0, 0.0, 1.0)) if n_ref else 0.0
    readiness = float(np.clip(0.45 * cap_score + 0.25 * ref_score + 0.3 * float(is_erp), 0.0, 1.0))
    out["dense360"] = {
        "arxiv_id": "2506.14471",
        "omni_vlm_readiness": round(readiness, 4),
        "caption_count": n_cap,
        "ref_count": n_ref,
        "is_erp": is_erp,
    }
    return out
