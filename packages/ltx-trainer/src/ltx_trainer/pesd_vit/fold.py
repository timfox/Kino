"""Fold PESD-ViT NAS multitask proxies into LTX video_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pesd_vit.activations import spatial_activation_map
from ltx_trainer.pesd_vit.config import PesdVitConfig
from ltx_trainer.pesd_vit.correlation import negative_transfer_risk, pearson_matrix
from ltx_trainer.pesd_vit.hooks import _nas_keywords


def pesd_vit_meta_block() -> dict[str, Any]:
    return {
        "pesd_vit": {
            "arxiv_id": "2605.29852",
            "fold_role": "nafld_nas_multitask_proxy",
            "tasks": ["steatosis", "ballooning", "inflammation"],
        }
    }


def _latent_task_scores(latents: np.ndarray) -> dict[str, float]:
    z = latents.reshape(latents.shape[0], -1).astype(np.float64)
    if z.shape[0] < 3:
        return {"steatosis": 0.5, "ballooning": 0.5, "inflammation": 0.5}
    # Toy ordinal proxies from channel statistics
    bright = float(np.percentile(z, 90))
    var = float(z.std())
    edge = float(np.abs(np.diff(z, axis=0)).mean()) if z.shape[0] > 1 else var
    return {
        "steatosis": float(np.clip(bright, 0, 1)),
        "ballooning": float(np.clip(var / (var + 1.0), 0, 1)),
        "inflammation": float(np.clip(edge / (edge + 0.1), 0, 1)),
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = PesdVitConfig()
    out = dict(data)
    out.update(pesd_vit_meta_block())

    cap = str(data.get("caption") or "")
    cues = _nas_keywords(cap)

    latents = data.get("latents")
    scores = {"steatosis": 0.5, "ballooning": 0.5, "inflammation": 0.5}
    heatmap_peak = 0.0
    if latents is not None:
        arr = np.asarray(
            latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
        )
        if arr.size > 0:
            scores = _latent_task_scores(arr)
            hm = spatial_activation_map(arr.ravel()[: 49 * 8])
            heatmap_peak = float(hm.max())

    score_vec = np.array([scores[t] for t in cfg.task_names], dtype=np.float64)
    # Synthetic mini-cohort from score proxies for correlation risk
    cohort = np.stack([score_vec, score_vec * 0.9 + 0.05, score_vec * 0.85 + 0.1])
    corr = pearson_matrix(cohort)

    out["pesd_vit"].update(
        {
            "caption_cues": cues,
            "task_score_proxy": {k: round(scores[k], 4) for k in scores},
            "ortho_lambda": cfg.ortho_lambda,
            "adapter_rank": cfg.adapter_rank,
            "heatmap_peak": round(heatmap_peak, 4),
            "corr_risk_proxy": round(negative_transfer_risk(corr), 4),
            "multitask_stable": heatmap_peak > 0.3 or any(cues.values()),
        }
    )
    return out
