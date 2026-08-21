"""Fold PanoFlight survey coverage metadata."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    task = str(meta.get("pano_task") or meta.get("vision_task") or "")
    known_tasks = {"detection", "segmentation", "depth", "vqa", "generation", "localization"}
    task_ok = any(t in task.lower() for t in known_tasks) if task else is_erp
    readiness = float(np.clip(0.55 * float(is_erp) + 0.45 * float(task_ok), 0.0, 1.0))
    out["pano_flight"] = {
        "arxiv_id": "2509.04444",
        "survey_task_readiness": round(readiness, 4),
        "is_erp": is_erp,
        "task": task or None,
    }
    return out
