"""Fold ID-LoRA identity / structured-prompt proxies into LTX latent shards."""

from __future__ import annotations

from typing import Any

from ltx_trainer.id_lora.config import IdLoraConfig
from ltx_trainer.id_lora.prompts import parse_id_lora_prompt


def id_lora_meta_block(cfg: IdLoraConfig | None = None) -> dict[str, Any]:
    c = cfg or IdLoraConfig()
    return {
        "id_lora": {
            "arxiv_id": c.paper_arxiv,
            "fold_role": "identity_av_ic_lora_proxy",
            "task": "identity_driven_av_personalization",
            "lora_rank": c.lora_rank,
        }
    }


def _visual_identity_score(visual: str) -> float:
    words = [w for w in visual.split() if w]
    if not words:
        return 0.0
    return min(1.0, len(words) / 8.0)


def _speech_identity_score(speech: str) -> float:
    words = [w for w in speech.split() if w]
    if not words:
        return 0.0
    return min(1.0, len(words) / 5.0)


def _identity_proxy(parsed: ParsedIdLoraPrompt) -> float:
    """Tag-aware readiness: structured [VISUAL]+[SPEECH] drives AV fold upweight."""
    if not parsed.identity_ready:
        return 0.25
    structured_bonus = 0.15 if parsed.is_structured else 0.0
    visual_score = _visual_identity_score(parsed.visual)
    speech_score = _speech_identity_score(parsed.speech)
    return min(1.0, structured_bonus + 0.30 * visual_score + 0.55 * speech_score)


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    """Video shard: structured prompt + first-frame / identity readiness."""
    c = IdLoraConfig()
    cap = str(data.get("caption") or data.get("prompt") or "")
    parsed = parse_id_lora_prompt(cap, c)
    identity_proxy = _identity_proxy(parsed)
    out = dict(data)
    out.update(id_lora_meta_block(c))
    out["id_lora"].update(
        {
            "has_visual_tag": bool(parsed.visual.strip()),
            "has_speech_tag": bool(parsed.speech.strip()),
            "has_sounds_tag": bool(parsed.sounds.strip()),
            "structured_prompt": parsed.is_structured,
            "identity_proxy": identity_proxy,
            "ref_audio_ready": parsed.identity_ready,
            "negative_ref_positions": c.use_negative_ref_positions,
        }
    )
    return out


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Audio shard: reference-audio IC slot readiness from caption speech tag."""
    c = IdLoraConfig()
    cap = str(data.get("caption") or data.get("prompt") or "")
    parsed = parse_id_lora_prompt(cap, c)
    speech_tokens = max(1, len((parsed.speech or "").split())) if parsed.speech else 0
    speaker_proxy = _speech_identity_score(parsed.speech) if parsed.speech else 0.0
    out = dict(data)
    out.update(id_lora_meta_block(c))
    out["id_lora"].update(
        {
            "ref_audio_slot_ready": bool(parsed.speech.strip()),
            "speech_token_estimate": speech_tokens,
            "speaker_identity_proxy": speaker_proxy,
            "identity_guidance_hint": c.default_identity_guidance,
        }
    )
    return out
