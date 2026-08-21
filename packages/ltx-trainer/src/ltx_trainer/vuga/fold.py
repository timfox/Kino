"""Fold VUGA viewport-unaware BOIQA proxy into video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.vuga.config import PAPER_ARXIV, VUGAConfig


def vuga_meta_block() -> dict[str, Any]:
    return {
        "vuga": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "erp_boiqa_proxy",
            "reference_metric": "JUFE_SRCC",
            "viewport_unaware": True,
        }
    }


def _erp_aspect_score(h: int, w: int) -> float:
    """ERP clips are often ~1:2 (H:W); reward near-panoramic layout."""
    if w <= 0:
        return 0.5
    ratio = h / w
    return float(np.exp(-((ratio - 0.5) ** 2) / 0.08))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = VUGAConfig()
    out = dict(data)
    out.update(vuga_meta_block())
    latents = data.get("latents")
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    if arr.ndim >= 4:
        # [C,F,H,W] or [F,C,H,W] — use last two dims as spatial
        spatial = arr.shape[-2:]
        h, w = int(spatial[0]), int(spatial[1])
    else:
        h, w = 8, 16

    spatial_std = float(arr.std())
    erp_prior = _erp_aspect_score(h, w)
    # Unified BOIQA proxy: lower latent noise + ERP layout → higher MOS (1–5)
    mos_proxy = float(np.clip(2.5 + erp_prior * 1.5 - spatial_std * 1.2, 1.0, 5.0))
    cmp_uniformity = float(np.clip(1.0 - spatial_std * 0.35, 0.0, 1.0))

    out["vuga"].update(
        {
            "mos_proxy": round(mos_proxy, 3),
            "cmp_uniformity_proxy": round(cmp_uniformity, 4),
            "erp_aspect_score": round(erp_prior, 4),
            "input_size": cfg.input_size,
        }
    )
    return out
