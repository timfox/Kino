"""Fold MTPano multi-task dense prediction readiness."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_erp = bool(meta.get("erp") or meta.get("is_360"))
    tasks = meta.get("mtpano_tasks") or meta.get("dense_tasks") or []
    n_tasks = len(tasks) if hasattr(tasks, "__len__") else int(meta.get("task_count") or 0)
    task_score = float(np.clip(n_tasks / 3.0, 0.0, 1.0)) if n_tasks else (0.55 if is_erp else 0.2)
    readiness = float(np.clip(0.5 * task_score + 0.5 * float(is_erp), 0.0, 1.0))
    out["mtpano"] = {
        "arxiv_id": "2602.05330",
        "multitask_dense_readiness": round(readiness, 4),
        "task_count": n_tasks,
        "is_erp": is_erp,
    }
    return out
