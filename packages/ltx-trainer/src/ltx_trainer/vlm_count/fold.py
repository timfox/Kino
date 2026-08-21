"""Fold VLM counting-bottleneck proxies into LTX video_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.vlm_count.config import VlmCountConfig
from ltx_trainer.vlm_count.hooks import _counting_keywords
from ltx_trainer.vlm_count.regimes import regime_for_n


def vlm_count_meta_block() -> dict[str, Any]:
    return {
        "vlm_count": {
            "arxiv_id": "2605.30170",
            "fold_role": "visual_counting_bottleneck_proxy",
        }
    }


def _density_proxy(latents: np.ndarray) -> float:
    z = latents.reshape(latents.shape[0], -1).astype(np.float64)
    if z.shape[0] < 1:
        return 0.0
    return float(np.clip(z.std(), 0, 1))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = VlmCountConfig()
    out = dict(data)
    out.update(vlm_count_meta_block())

    cap = str(data.get("caption") or "")
    cues = _counting_keywords(cap)
    n_hint = cues.get("numeric_hint")
    regime = regime_for_n(int(n_hint), visual_train_max=cfg.visual_train_max) if n_hint else "ID"

    latents = data.get("latents")
    density = 0.0
    if latents is not None:
        arr = np.asarray(
            latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
        )
        if arr.size > 0:
            density = _density_proxy(arr)

    out["vlm_count"].update(
        {
            "caption_cues": {k: v for k, v in cues.items() if k != "numeric_hint"},
            "numeric_hint": n_hint,
            "regime_proxy": regime,
            "latent_density_proxy": round(density, 4),
            "symbolic_mapping_risk": regime in ("VE", "FE") or cues.get("counting_cue", False),
            "enumerable_scene": cues.get("enumerable_objects", False) or cues.get("grid_or_board", False),
        }
    )
    return out
