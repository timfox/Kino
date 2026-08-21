"""Fold SemanticStitch seam coherence into proceduralsky ERP latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360") or hdr_meta.get("projection") == "erp")
    stitched = bool(meta.get("semantic_stitch_merged") or meta.get("stitched_erp"))
    overlap = int(meta.get("overlap_captures") or meta.get("bracket_count") or 0)
    salient_ok = bool(meta.get("salient_seam_clear") or meta.get("sun_disk_preserved"))
    overlap_score = float(np.clip(overlap / 3.0, 0.0, 1.0)) if overlap else (0.4 if stitched else 0.1)
    salient_score = 0.35 if salient_ok else (0.2 if stitched else 0.05)
    erp_score = 0.25 if is_erp else 0.05
    coherence = float(np.clip(erp_score + overlap_score + salient_score, 0.0, 1.0))
    out["semantic_stitch"] = {
        "arxiv_id": "2511.12084",
        "seam_coherence": round(coherence, 4),
        "is_erp": is_erp,
        "stitched_erp": stitched,
        "overlap_captures": overlap,
        "salient_seam_clear": salient_ok,
    }
    return out
