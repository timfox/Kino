"""SVHighlights TF-SELECTOR saliency proxies on video latent shards (arXiv:2606.06926)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ltx_trainer.svhighlights.config import SvHighlightsConfig
from ltx_trainer.svhighlights.tf_selector import llm_saliency_stub

_HIGHLIGHT_CUES = re.compile(
    r"\b(goal|score|touchdown|celebration|crowd|winning|fast break|home run|"
    r"checkered flag|highlight|slam dunk|touchdown|strikeout)\b",
    re.I,
)
_LOW_CUES = re.compile(r"\b(timeout|commercial|halftime|intro|interview|quiet)\b", re.I)


def svhighlights_meta_block(cfg: SvHighlightsConfig | None = None) -> dict[str, Any]:
    c = cfg or SvHighlightsConfig()
    return {
        "svhighlights": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "tf_selector_saliency_proxy",
            "segment_scoring": "caption+transcript+volume_stub",
        }
    }


def _caption_text(data: dict[str, Any]) -> str:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    for key in ("caption", "prompt", "text", "description", "transcript"):
        val = meta.get(key) if isinstance(meta, dict) else None
        if isinstance(val, str) and val.strip():
            return val.strip()
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _latent_motion_energy(latents: Any) -> float:
    if latents is None:
        return 0.35
    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 8:
        return 0.2
    return float(np.clip(arr.std() / (np.abs(flat).mean() + 1e-6), 0.05, 1.2))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach svhighlights.* sidecars for long-form highlight saliency weighting."""
    c = SvHighlightsConfig()
    cap = _caption_text(data)
    motion = _latent_motion_energy(data.get("latents"))
    volume = float(np.clip(0.25 + 0.5 * min(motion, 1.0), 0.0, 1.0))
    hi_hits = len(_HIGHLIGHT_CUES.findall(cap))
    low_hits = len(_LOW_CUES.findall(cap))
    saliency = llm_saliency_stub(cap, cap, volume, cfg=c)
    if hi_hits:
        saliency = float(np.clip(saliency + 0.15 * min(hi_hits, 3), 0.0, c.saliency_max))
    if low_hits:
        saliency = float(np.clip(saliency - 0.2 * min(low_hits, 2), c.saliency_min, c.saliency_max))
    norm = float(np.clip(saliency / c.saliency_max, 0.0, 1.0))
    highlight_ready = norm >= 0.45 and hi_hits > 0
    out = dict(data)
    out.update(svhighlights_meta_block(c))
    out["svhighlights"].update(
        {
            "segment_saliency": round(saliency, 4),
            "saliency_norm": round(norm, 4),
            "highlight_cue_hits": hi_hits,
            "low_cue_hits": low_hits,
            "motion_energy_proxy": round(motion, 4),
            "volume_proxy": round(volume, 4),
            "highlight_ready": highlight_ready,
            "recap_weight_proxy": round(0.35 + 0.65 * norm, 4),
        }
    )
    return out
