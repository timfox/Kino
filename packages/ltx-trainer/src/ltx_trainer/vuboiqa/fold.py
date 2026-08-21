"""Fold VU-BOIQA viewport-unaware OIQA proxy into video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.vuboiqa.config import PAPER_ARXIV, VuBoiqaConfig


def vuboiqa_meta_block() -> dict[str, Any]:
    return {
        "vuboiqa": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "erp_boiqa_proxy",
            "reference_metric": "JUFE_SRCC",
            "viewport_unaware": True,
        }
    }


def _patch_diversity(arr: np.ndarray, *, num_patches: int) -> float:
    """APS-style patch diversity proxy on latent spatial slices."""
    if arr.ndim < 4:
        return 0.5
    z = arr
    if z.shape[0] <= 64 and z.ndim == 4:
        # [F,C,H,W] or [C,F,H,W] — use frame/slice axis 0
        slices = z[: min(num_patches, z.shape[0])]
    else:
        slices = z.reshape(-1, *z.shape[-2:])[:num_patches]
    if slices.shape[0] < 2:
        return 0.5
    flat = slices.reshape(slices.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    normed = flat / norms
    sim = normed @ normed.T
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    mean_sim = float(off.mean()) if off.size else 0.5
    return float(np.clip(1.0 - mean_sim, 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = VuBoiqaConfig()
    out = dict(data)
    out.update(vuboiqa_meta_block())
    latents = data.get("latents")
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    spatial_std = float(arr.std())
    diversity = _patch_diversity(arr, num_patches=cfg.num_patches)
    quality_proxy = float(np.clip(2.0 + diversity * 2.0 - spatial_std, 1.0, 5.0))
    lgqa_stability = float(np.clip(1.0 - spatial_std * 0.4, 0.0, 1.0))

    out["vuboiqa"].update(
        {
            "quality_proxy": round(quality_proxy, 3),
            "mos_proxy": round(quality_proxy, 3),
            "patch_diversity_proxy": round(diversity, 4),
            "lgqa_stability_proxy": round(lgqa_stability, 4),
            "num_patches": cfg.num_patches,
        }
    )
    return out
