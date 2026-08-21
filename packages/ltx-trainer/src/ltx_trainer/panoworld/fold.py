"""Fold PanoWorld spherical MLLM readiness into ERP latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    patches = int(meta.get("num_patches") or meta.get("panoworld_patches") or 0)
    has_instr = bool(meta.get("instruction") or meta.get("panoworld_task"))
    patch_score = float(np.clip(patches / 64.0, 0.0, 1.0)) if patches else (0.6 if is_erp else 0.2)
    readiness = float(np.clip(0.45 * patch_score + 0.35 * float(is_erp) + (0.2 if has_instr else 0.0), 0.0, 1.0))
    out["panoworld"] = {
        "arxiv_id": "2605.13169",
        "panospace_readiness": round(readiness, 4),
        "is_erp": is_erp,
        "num_patches": patches,
    }
    return out
