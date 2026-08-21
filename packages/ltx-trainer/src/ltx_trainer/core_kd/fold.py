"""Fold CoRe-KD MER proxies into LTX video_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.core_kd.config import CoreKDConfig
from ltx_trainer.core_kd.hooks import _mer_keywords
from ltx_trainer.core_kd.poe import poe_fuse


def core_kd_meta_block() -> dict[str, Any]:
    return {
        "core_kd": {
            "arxiv_id": "2605.29590",
            "fold_role": "conversational_mer_missing_modality_proxy",
        }
    }


def _latent_modality_proxies(latents: np.ndarray) -> dict[str, float]:
    z = latents.reshape(latents.shape[0], -1).astype(np.float64)
    if z.shape[0] < 2:
        return {"text": 0.5, "audio": 0.5, "video": 0.5}
    frame_n = np.linalg.norm(z, axis=1)
    return {
        "text": float(np.clip(frame_n.mean(), 0, 1)),
        "audio": float(np.clip(frame_n.std(), 0, 1)),
        "video": float(np.clip(np.percentile(frame_n, 90), 0, 1)),
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = CoreKDConfig()
    out = dict(data)
    out.update(core_kd_meta_block())

    cap = str(data.get("caption") or "")
    cues = _mer_keywords(cap)

    latents = data.get("latents")
    proxies = {"text": 0.5, "audio": 0.5, "video": 0.5}
    state_drift_proxy = 0.0
    if latents is not None:
        arr = np.asarray(
            latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
        )
        if arr.size > 0:
            proxies = _latent_modality_proxies(arr)
            mus = [np.full(cfg.state_dim, proxies[m]) for m in cfg.modalities]
            precs = [np.ones(cfg.state_dim) * (0.5 + proxies[m]) for m in cfg.modalities]
            fused_mu, fused_sigma = poe_fuse(mus, precs)
            teacher_mu = np.ones(cfg.state_dim) * 0.5
            state_drift_proxy = float(np.mean((fused_mu - teacher_mu) ** 2 + fused_sigma**2))

    out["core_kd"].update(
        {
            "caption_cues": cues,
            "modality_proxy": {k: round(proxies[k], 4) for k in proxies},
            "state_drift_proxy": round(state_drift_proxy, 4),
            "missing_modality_ready": cues.get("missing_modality", False) or state_drift_proxy > 0.01,
            "conflict_regularized": cues.get("modality_conflict", False),
        }
    )
    return out
