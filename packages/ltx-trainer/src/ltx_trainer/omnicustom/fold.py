"""Fold OmniCustom sync AV customization proxies into LTX latent shards."""

from __future__ import annotations

from typing import Any

from ltx_trainer.omnicustom.config import OmniCustomConfig
from ltx_trainer.omnicustom.prompts import extract_speech


def omnicustom_meta_block(cfg: OmniCustomConfig | None = None) -> dict[str, Any]:
    c = cfg or OmniCustomConfig()
    return {
        "omnicustom": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "sync_av_customization_proxy",
            "task": "sync_audio_video_customization",
        }
    }


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach identity / speech-span readiness from caption + latent shard."""
    c = OmniCustomConfig()
    cap = str(data.get("caption") or data.get("prompt") or "")
    parsed = extract_speech(cap, c)
    out = dict(data)
    out.update(omnicustom_meta_block(c))
    out["omnicustom"].update(
        {
            "has_speech_span": parsed.has_speech,
            "speech_chars": len(parsed.speech_text or ""),
            "sync_custom_ready": parsed.has_speech and bool(cap.strip()),
            "identity_proxy": 1.0 if parsed.has_speech else 0.25,
            "lora_rank": c.lora_rank,
            "ref_audio_seconds": c.ref_audio_seconds,
        }
    )
    return out


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach timbre-slot proxies from caption on audio latent shards."""
    c = OmniCustomConfig()
    cap = str(data.get("caption") or data.get("prompt") or "")
    parsed = extract_speech(cap, c)
    tokens = max(1, len((parsed.speech_text or "").split()))
    out = dict(data)
    out.update(omnicustom_meta_block(c))
    out["omnicustom"].update(
        {
            "timbre_slot_ready": True,
            "speech_token_estimate": tokens,
            "timbre_proxy": min(1.0, tokens / 20.0),
            "has_speech_span": parsed.has_speech,
        }
    )
    return out
