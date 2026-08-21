"""Fold StEM2 cinema-mapping readiness into HDR latent sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.stem2_sdr_hdr.pipeline import isotonic_readiness_from_triplet


def stem2_meta_block() -> dict[str, Any]:
    return {
        "stem2_sdr_hdr": {
            "arxiv_id": "2604.06276",
            "dataset": "ASC StEM2",
            "mapping_model": "global_isotonic + sparse_residuals",
            "fold_role": "cinema_mastering_sidecar",
        }
    }


def _luminance_std(rgb: np.ndarray) -> float:
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        return float(np.std(y))
    if rgb.ndim == 4 and rgb.shape[1] == 3:
        y = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
        return float(np.std(y))
    return float(np.std(rgb))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(stem2_meta_block())

    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    scene = str(hdr_meta.get("stem2_scene") or hdr_meta.get("scene_type") or "desert")

    readiness = isotonic_readiness_from_triplet(scene, seed=abs(hash(scene)) % 10_000)
    luma_spread = 0.0

    for key in ("hdr_linear", "sdr_preview"):
        val = data.get(key)
        if val is None:
            continue
        arr = np.asarray(val.detach().cpu().float().numpy() if hasattr(val, "detach") else val)
        if arr.size:
            luma_spread = _luminance_std(arr)
            break
    else:
        latents = data.get("latents")
        if latents is not None:
            arr = np.asarray(
                latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
            )
            luma_spread = float(np.std(arr))

    out["stem2_sdr_hdr"].update(
        {
            "scene_proxy": scene,
            "isotonic_readiness_proxy": round(readiness, 4),
            "luminance_spread_proxy": round(luma_spread, 5),
            "exr_closer_prior": 0.824,
            "hdr_vae_encoding": hdr_meta.get("hdr_vae_encoding"),
        }
    )
    return out
