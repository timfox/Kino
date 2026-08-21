"""EvoGS fold sidecar: progressive LOD + ψ-energy proxy on video latents."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ltx_trainer.evogs.config import EvoGSConfig
from ltx_trainer.evogs.metrics import ghost_splat_ratio


def evogs_meta_block(cfg: EvoGSConfig | None = None) -> dict[str, Any]:
    c = cfg or EvoGSConfig()
    return {
        "evogs": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "continuous_layer_lod_proxy",
            "levels": c.params.num_levels,
        }
    }


def _motion_energy(latents: Any) -> float:
    if latents is None:
        return 0.3
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 8:
        return 0.2
    return float(np.clip(arr.std() / (np.abs(flat).mean() + 1e-6), 0.05, 1.2))


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


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach evogs.* sidecars for progressive streaming weighting."""
    c = EvoGSConfig()
    motion = _motion_energy(data.get("latents"))
    cap = _caption_text(data)
    detail_cues = len(re.findall(r"\b(close|detail|texture|fast|action|highlight)\b", cap, re.I))
    psi_energy = float(np.clip(0.1 + 0.6 * motion + 0.05 * detail_cues, 0.0, 1.0))
    lod = int(np.clip(np.floor(psi_energy * c.params.num_levels), 0, c.params.num_levels - 1))
    opacity_proxy = float(np.clip(0.35 + 0.5 * (1.0 - motion), 0.01, 0.95))
    ghost = ghost_splat_ratio(np.array([opacity_proxy]), c.params.opacity_ghost_threshold)
    out = dict(data)
    out.update(evogs_meta_block(c))
    out["evogs"].update(
        {
            "lod_level": lod,
            "psi_energy_norm": round(psi_energy, 4),
            "ghost_ratio_proxy": round(ghost, 4),
            "storage_mb_proxy": round(0.02 + 0.15 * psi_energy * (lod + 1), 4),
            "mem_mb_proxy": round(0.015 + 0.12 * psi_energy, 4),
            "stream_priority": round(psi_energy, 4),
        }
    )
    return out
