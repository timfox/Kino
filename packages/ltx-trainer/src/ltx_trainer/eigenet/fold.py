"""Fold EIGENET few-shot RIR proxies into LTX audio_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.eigenet.config import EigeNetConfig
from ltx_trainer.eigenet.hooks import score_spatial_audio_caption


def eigenet_meta_block() -> dict[str, Any]:
    return {
        "eigenet": {
            "arxiv_id": "2605.28101",
            "fold_role": "few_shot_novel_view_rir_proxy",
            "task": "novel_view_rir_prediction",
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach reverberation / geometry-readiness proxies from latent statistics."""
    cfg = EigeNetConfig()
    latents = data.get("latents")
    caption = str(data.get("caption") or data.get("prompt") or "")
    out = dict(data)
    out.update(eigenet_meta_block())

    cap_score = score_spatial_audio_caption(caption) if caption else {"spatial_acoustic_cue_count": 0}
    k_ref = int(data.get("reference_views") or data.get("k_reference") or 1)
    k_ref = max(1, min(k_ref, cfg.n_reference_views_max))

    if latents is None:
        out["eigenet"].update({"k_reference": k_ref, **cap_score})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 8:
        flat = np.pad(flat, (0, 8 - flat.size))
    # Late-tail energy proxy for reverberation
    tail = flat[len(flat) // 2 :]
    early = flat[: len(flat) // 4]
    tail_e = float(np.mean(np.abs(tail)) + 1e-8)
    early_e = float(np.mean(np.abs(early)) + 1e-8)
    decay_ratio = float(np.clip(tail_e / early_e, 0.0, 4.0))
    temporal_var = float(np.std(flat))
    geometry_informed_ready = cap_score.get("few_shot_ready", False) or decay_ratio > 0.15

    out["eigenet"].update(
        {
            "k_reference": k_ref,
            "decay_ratio_proxy": round(decay_ratio, 4),
            "temporal_var_proxy": round(temporal_var, 4),
            "geometry_informed_ready": geometry_informed_ready,
            "n_acoustic_tokens_ref": cfg.n_acoustic_tokens,
            **cap_score,
        }
    )
    return out
