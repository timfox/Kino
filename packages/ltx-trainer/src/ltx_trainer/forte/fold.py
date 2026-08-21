"""FORTE T2A alignment proxies on video / audio latent shards (arXiv:2606.05812)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.forte.config import ForteConfig
from ltx_trainer.forte.fol import parse_query_fallback, pred_set, predicate_overlap
from ltx_trainer.forte.prompt_bridge import refine_prompt_for_ltx
from ltx_trainer.forte.rerank import caption_to_fol


def forte_meta_block(cfg: ForteConfig | None = None) -> dict[str, Any]:
    c = cfg or ForteConfig()
    return {
        "forte": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "fol_t2a_alignment_proxy",
            "stage": "alignment_rerank_sidecar",
        }
    }


def forte_audio_meta_block(cfg: ForteConfig | None = None) -> dict[str, Any]:
    c = cfg or ForteConfig()
    return {
        "forte_audio": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "fol_audio_latent_align",
            "stage": "stage2_projection_proxy",
        }
    }


def _caption_text(data: dict[str, Any]) -> str:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    for key in ("caption", "prompt", "text", "description"):
        val = meta.get(key) if isinstance(meta, dict) else None
        if isinstance(val, str) and val.strip():
            return val.strip()
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _latent_spectral_energy(latents: Any) -> float:
    if latents is None:
        return 0.5
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 8:
        return 0.05
    return float(np.clip(arr.std() / (np.abs(flat).mean() + 1e-6), 0.02, 1.5))


def _fol_sidecar_fields(caption: str, latents: Any, *, cfg: ForteConfig) -> dict[str, Any]:
    caption = caption.strip()
    if not caption:
        return {
            "t2a_align_proxy": 0.45,
            "fol_predicate_count": 0,
            "caption_fol_overlap": 0.0,
            "rerank_blend_proxy": 0.45,
            "q_star": "",
        }
    refined = refine_prompt_for_ltx(caption, cfg=cfg, anchor_caption=caption)
    phi_c = caption_to_fol(caption)
    n_pred = len(pred_set(phi_c))
    spectral = _latent_spectral_energy(latents)
    overlap = float(refined.get("caption_fol_overlap") or predicate_overlap(parse_query_fallback(refined["q_star"]), phi_c))
    richness = min(1.0, 0.15 + 0.08 * n_pred)
    align = float(np.clip(0.35 + 0.45 * overlap + 0.2 * min(spectral, 1.0) + richness * 0.15, 0.0, 1.0))
    rerank = float(np.clip(cfg.alpha_rerank * overlap + (1.0 - cfg.alpha_rerank) * align, 0.0, 1.0))
    return {
        "t2a_align_proxy": round(align, 4),
        "fol_predicate_count": n_pred,
        "caption_fol_overlap": round(overlap, 4),
        "rerank_blend_proxy": round(rerank, 4),
        "q_star": refined["q_star"],
        "refinement_score": refined["score"],
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Video latent shard: caption FOL + cross-modal T2A readiness proxy."""
    cfg = ForteConfig()
    out = dict(data)
    out.update(forte_meta_block(cfg))
    caption = _caption_text(data)
    fields = _fol_sidecar_fields(caption, data.get("latents"), cfg=cfg)
    out["forte"].update(fields)
    return out


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Audio latent shard: Stage-2/3 alignment proxy from caption + spectral stats."""
    cfg = ForteConfig()
    out = dict(data)
    out.update(forte_audio_meta_block(cfg))
    caption = _caption_text(data)
    fields = _fol_sidecar_fields(caption, data.get("latents"), cfg=cfg)
    out["forte_audio"].update(fields)
    out["forte_audio"]["projection_ready"] = fields["t2a_align_proxy"] > 0.5
    return out
