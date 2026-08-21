"""AV-fold sidecar: flood extent + SWE physics proxies for LTX training."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.flood_physics.constants import PAPER_ARXIV


def flood_physics_meta_block() -> dict[str, Any]:
    return {
        "flood_physics": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "physics_guided_flood_hydrology",
            "method": "unet_fno_swe_residuals",
        }
    }


def _latent_water_extent_proxy(arr: np.ndarray) -> float:
    """Low-frequency / smooth regions proxy inundation extent."""
    if arr.ndim >= 3:
        frame = arr[0] if arr.ndim == 4 else arr
        if frame.ndim == 3:
            frame = frame.mean(axis=0)
    else:
        frame = arr
    smooth = float(np.std(frame.astype(np.float64)))
    low_freq = float(np.mean(np.abs(frame)))
    return float(np.clip(0.5 * (1.0 - smooth) + 0.5 * low_freq, 0.0, 1.0))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg_block = flood_physics_meta_block()
    out = dict(data)
    out.update(cfg_block)

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("flood", "inundation", "sentinel-1", "sar flood", "floodplain")):
        regime = "flood_event"
    elif any(w in caption for w in ("river", "hydrology", "water depth", "inundated")):
        regime = "hydrology"
    elif any(w in caption for w in ("sentinel-2", "ndwi", "optical water", "wetland")):
        regime = "optical_water"
    elif any(w in caption for w in ("dem", "hand", "terrain", "topography")):
        regime = "terrain_context"
    else:
        regime = "unknown_hydrology"

    latents = data.get("latents")
    if latents is None:
        out["flood_physics"].update({"regime_hint": regime, "has_latents": False})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    extent = _latent_water_extent_proxy(arr)
    spatial_std = float(arr.std())
    physics_consistency = float(np.clip(1.0 - abs(spatial_std - 0.35), 0.0, 1.0))
    inundation_readiness = float(np.clip(extent * (0.4 + 0.6 * physics_consistency), 0.0, 1.0))

    out["flood_physics"].update(
        {
            "regime_hint": regime,
            "has_latents": True,
            "flood_extent_proxy": round(extent, 4),
            "physics_consistency_proxy": round(physics_consistency, 4),
            "inundation_readiness_proxy": round(inundation_readiness, 4),
            "modalities": ["sentinel1_sar", "sentinel2_optical", "dem_terrain"],
        }
    )
    return out
