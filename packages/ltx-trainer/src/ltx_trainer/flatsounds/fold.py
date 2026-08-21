"""Fold FlatSounds V2A physical proxies into LTX video_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.flatsounds.config import FlatSoundsConfig
from ltx_trainer.flatsounds.hooks import _impact_keywords


def flatsounds_meta_block() -> dict[str, Any]:
    return {
        "flatsounds": {
            "arxiv_id": "2605.30339",
            "fold_role": "v2a_physical_benchmark_proxy",
        }
    }


def _latent_impact_proxy(latents: np.ndarray) -> dict[str, float]:
    z = latents.reshape(latents.shape[0], -1).astype(np.float64)
    if z.shape[0] < 2:
        return {"onset_strength": 0.0, "temporal_modulation": 0.0}
    frame_energy = np.linalg.norm(z, axis=1)
    diff = np.abs(np.diff(frame_energy))
    return {
        "onset_strength": float(np.percentile(diff, 95)),
        "temporal_modulation": float(diff.std() / (diff.mean() + 1e-9)),
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = FlatSoundsConfig()
    out = dict(data)
    out.update(flatsounds_meta_block())

    cap = str(data.get("caption") or "")
    cues = _impact_keywords(cap)

    latents = data.get("latents")
    proxy = {"onset_strength": 0.0, "temporal_modulation": 0.0}
    if latents is not None:
        arr = np.asarray(
            latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
        )
        if arr.size > 0:
            proxy = _latent_impact_proxy(arr)

    out["flatsounds"].update(
        {
            "caption_cues": cues,
            "onset_strength_proxy": round(proxy["onset_strength"], 4),
            "temporal_modulation_proxy": round(proxy["temporal_modulation"], 4),
            "dataset_clips": cfg.dataset_clips,
            "physics_test_cases": cfg.physics_test_cases,
            "impact_timing_ready": proxy["onset_strength"] > 0.01 or cues.get("impact_cue", False),
            "physics_audit_ready": any(cues.values()) or proxy["temporal_modulation"] > 0.05,
        }
    )
    return out
