"""Audio latent sidecar: SwanSphere spatial / streaming proxies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.swansphere.config import SwanSphereConfig


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    cfg = SwanSphereConfig()
    enc = data.get("audio_encode") if isinstance(data.get("audio_encode"), dict) else {}
    prep = enc.get("prep") if isinstance(enc.get("prep"), dict) else {}
    layout = prep.get("input_layout") or data.get("input_layout")
    latent_fps = float(enc.get("latent_fps") or 0.0)
    ref_fps = cfg.latent_fps
    fps_ratio = latent_fps / ref_fps if ref_fps > 0 and latent_fps > 0 else 1.0
    latents = data.get("latents")
    energy = 0.0
    if latents is not None:
        arr = np.asarray(
            latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
        ).astype(np.float64)
        energy = float(np.mean(arr**2))
    out["swansphere_audio"] = {
        "arxiv": cfg.paper_arxiv,
        "fold_role": "streaming_foa_proxy",
        "layout": layout,
        "has_foa": layout == "foa_wxyz" or bool(prep.get("foa_downmix")),
        "foa_azimuth_rad": prep.get("foa_azimuth_rad"),
        "latent_fps": enc.get("latent_fps"),
        "reference_latent_fps": ref_fps,
        "fps_ratio_vs_swan": round(fps_ratio, 4),
        "sync_offset_frames": enc.get("sync_offset_frames"),
        "streaming_ready": abs(fps_ratio - 1.0) < 0.15,
        "latent_energy": round(energy, 6),
        "first_chunk_latency_s": cfg.first_chunk_latency_s,
    }
    return out
