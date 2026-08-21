"""AV-fold sidecar: rank-one LoRA / attention pre-train readiness proxies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lora_hd_attn.constants import PAPER_ARXIV


def lora_hd_attn_meta_block() -> dict[str, Any]:
    return {
        "lora_hd_attn": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "rank_one_lora_hd_theory",
            "method": "effective_noise_attention_lora",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(lora_hd_attn_meta_block())

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    caption = str(
        meta.get("caption") or meta.get("prompt") or meta.get("text") or data.get("caption") or ""
    ).lower()
    if any(w in caption for w in ("lora", "low-rank adaptation", "rank-one", "rank one")):
        regime = "rank_one_lora"
    elif any(w in caption for w in ("attention pretrain", "pre-training attention", "aim model")):
        regime = "attention_pretrain"
    elif any(w in caption for w in ("reused sequence", "same sequences", "memorization")):
        regime = "reused_sequences"
    elif any(w in caption for w in ("active fine-tuning", "active finetuning", "sample selection")):
        regime = "active_finetuning"
    elif any(w in caption for w in ("softmax attention", "single-head attention", "transformer layer")):
        regime = "attention_layer"
    else:
        regime = "unknown_attention"

    latents = data.get("latents")
    if latents is None:
        out["lora_hd_attn"].update({"regime_hint": regime, "has_latents": False})
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    temporal = float(np.clip(1.0 / (1.0 + np.var(np.diff(arr.astype(np.float64), axis=0))), 0.0, 1.0))
    rank_one = float(np.clip(np.std(arr) / (np.abs(arr.mean()) + 1e-8), 0.0, 1.0))
    noise_proxy = float(np.clip(1.0 - temporal, 0.0, 1.0))
    readiness = float(np.clip(0.5 * temporal + 0.5 * (1.0 - noise_proxy), 0.0, 1.0))
    mismatch = float(np.clip(rank_one * noise_proxy, 0.0, 1.0))

    out["lora_hd_attn"].update(
        {
            "regime_hint": regime,
            "has_latents": True,
            "rank_one_alignment_proxy": round(rank_one, 4),
            "effective_noise_proxy": round(noise_proxy, 4),
            "lora_readiness_proxy": round(readiness, 4),
            "test_overlap_mismatch_proxy": round(mismatch, 4),
        }
    )
    return out
