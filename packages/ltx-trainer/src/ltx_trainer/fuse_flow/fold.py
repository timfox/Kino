"""Fold FUSE-Flow geometry / fusion readiness into video latent shards (arXiv:2602.01035)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fuse_flow.conditioning import latent_geometry_conditioning
from ltx_trainer.fuse_flow.geometry import (
    fusion_readiness_proxy,
    geometry_stability_proxy,
    multi_view_hint_from_meta,
)


def fuse_flow_meta_block() -> dict[str, Any]:
    return {
        "fuse_flow": {
            "arxiv_id": "2602.01035",
            "fold_role": "metric_multi_view_geometry",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(fuse_flow_meta_block())
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    if isinstance(data.get("caption"), str) and data["caption"].strip():
        meta = {**meta, "caption": data["caption"].strip()}
    path = str(
        meta.get("source_path")
        or meta.get("video_path")
        or meta.get("path")
        or data.get("relative_path")
        or ""
    )
    caption = str(meta.get("caption") or meta.get("prompt") or data.get("caption") or "")
    multi_view, n_cameras = multi_view_hint_from_meta(meta, path=path, caption=caption)

    latents = data.get("latents")
    if latents is None:
        out["fuse_flow"].update(
            {
                "geometry_stability_proxy": 0.5,
                "fusion_readiness_proxy": 0.45,
                "multi_view_hint": multi_view,
                "n_cameras_hint": n_cameras,
            }
        )
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    geom, jerk = geometry_stability_proxy(arr)
    fusion = fusion_readiness_proxy(arr)
    if multi_view:
        fusion = float(np.clip(fusion + 0.08, 0.0, 1.0))
    block = {
        "geometry_stability_proxy": round(geom, 4),
        "fusion_readiness_proxy": round(fusion, 4),
        "temporal_jerk_std": round(jerk, 5),
        "multi_view_hint": multi_view,
        "n_cameras_hint": n_cameras,
    }
    block["conditioning"] = latent_geometry_conditioning(arr)
    out["fuse_flow"].update(block)
    return out
