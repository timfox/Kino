"""Fold SphereUFormer icosphere perception readiness into ERP latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    is_360 = bool(meta.get("is_360") or meta.get("erp") or meta.get("projection") == "erp")
    task = str(meta.get("sphere_task") or meta.get("task") or "depth")
    nodes = int(meta.get("icosphere_nodes") or meta.get("hex_nodes") or 0)
    node_score = float(np.clip(nodes / 642.0, 0.0, 1.0)) if nodes else (0.7 if is_360 else 0.25)
    task_ok = task in ("depth", "segmentation", "seg")
    readiness = float(np.clip(0.5 * node_score + (0.35 if is_360 else 0.1) + (0.15 if task_ok else 0.0), 0.0, 1.0))
    out["sphereuformer"] = {
        "arxiv_id": "2412.06968",
        "perception_readiness": round(readiness, 4),
        "is_360": is_360,
        "task": task,
        "icosphere_nodes": nodes,
    }
    return out
