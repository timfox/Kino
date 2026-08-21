"""Fold EntroAD ZSAD proxies into LTX video_latents shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.entroad.config import EntroADConfig
from ltx_trainer.entroad.entropy import structural_entropy_map
from ltx_trainer.entroad.hooks import score_anomaly_caption


def entroad_meta_block() -> dict[str, Any]:
    return {
        "entroad": {
            "arxiv_id": "2605.28630",
            "fold_role": "zero_shot_anomaly_proxy",
            "task": "ZSAD",
        }
    }


def _pseudo_attention_from_latents(latents: np.ndarray) -> np.ndarray:
    """Toy patch graph from latent cosine similarities (routing prior proxy)."""
    z = latents.reshape(latents.shape[0], -1).astype(np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    sim = z @ z.T
    sim = np.clip(sim, 0.0, None)
    return sim


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach structural-entropy and branch-hint proxies per shard."""
    cfg = EntroADConfig()
    latents = data.get("latents")
    caption = str(data.get("caption") or data.get("prompt") or "")
    out = dict(data)
    out.update(entroad_meta_block())

    cap = score_anomaly_caption(caption) if caption else {"branch_prior_hint": "branch_a"}

    if latents is None:
        out["entroad"].update(cap)
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    if arr.ndim < 2:
        out["entroad"].update({"entropy_mean": None, **cap})
        return out

    n_patches = min(arr.shape[0], 64)
    sub = arr[:n_patches]
    attn = _pseudo_attention_from_latents(sub)
    e = structural_entropy_map(attn[None, ...], drop_cls=False)
    e_mean = float(e.mean())
    e_std = float(e.std())
    localized_risk = float(np.clip(e_std / (e_mean + 1e-6), 0.0, 2.0))

    out["entroad"].update(
        {
            "entropy_mean": round(e_mean, 4),
            "entropy_std": round(e_std, 4),
            "localized_entropy_spike_proxy": round(localized_risk, 4),
            "branch_prior_hint": cap.get("branch_prior_hint", "branch_a"),
            "fusion_alpha_ref": cfg.fusion_alpha,
            "zsad_ready": localized_risk > 0.2 or cap.get("anomaly_caption_ready", False),
            **cap,
        }
    )
    return out
