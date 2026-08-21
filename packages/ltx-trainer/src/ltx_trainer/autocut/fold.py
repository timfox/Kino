"""LTX fold sidecars — ad-editing readiness proxies from captions."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.autocut.config import AutoCutConfig

_PRODUCT_CUES = re.compile(
    r"\b(product|brand|buy|sale|offer|discount|features?|earbuds|cream|wireless|"
    r"subscribe|order|shop|limited)\b",
    re.I,
)
_SCRIPT_STRUCTURE = re.compile(r"[.!?]\s+")


def autocut_meta_block(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    return {
        "autocut": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "ad_video_editing_proxy",
            "tasks": "selection,sort,script,bgm",
        }
    }


def _caption_stats(cap: str) -> tuple[list[str], int, int]:
    sentences = [s.strip() for s in _SCRIPT_STRUCTURE.split(cap) if s.strip()]
    product_hits = len(_PRODUCT_CUES.findall(cap))
    clip_proxy = max(1, min(24, len(sentences) if sentences else 1))
    return sentences, product_hits, clip_proxy


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach autocut.* sidecars for advertisement editing training hints."""
    c = AutoCutConfig()
    cap = str(data.get("caption") or data.get("text") or data.get("prompt") or "")
    sentences, product_hits, clip_proxy = _caption_stats(cap)
    ad_ready = product_hits > 0 and len(cap) > 20
    selection_ready = clip_proxy >= 2
    narrative = min(1.0, len(sentences) / 6.0) * (0.85 if product_hits else 0.35)
    out = dict(data)
    out.update(autocut_meta_block(c))
    out["autocut"].update(
        {
            "has_product_language": product_hits > 0,
            "sentence_count_proxy": len(sentences),
            "clip_count_proxy": clip_proxy,
            "ad_edit_ready": ad_ready,
            "selection_task_ready": selection_ready,
            "edit_ready_proxy": 1.0 if ad_ready else 0.2,
            "narrative_proxy": round(float(narrative), 4),
            "selection_proxy": round(min(1.0, clip_proxy / 8.0) if selection_ready else 0.3, 4),
        }
    )
    return out


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    cap = str(data.get("caption") or data.get("prompt") or "")
    speech_present = bool(cap.strip())
    c = AutoCutConfig()
    out = dict(data)
    out.update(autocut_meta_block(c))
    bgm_proxy = min(1.0, len(cap) / 60.0) if speech_present else 0.35
    out["autocut"].update(
        {
            "bgm_slot_ready": True,
            "speech_present": speech_present,
            "mss_eval_ready": len(cap) > 10,
            "bgm_proxy": round(float(bgm_proxy), 4),
        }
    )
    return out
