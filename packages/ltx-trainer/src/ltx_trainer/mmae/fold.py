"""AV-fold sidecar: MMAE taxonomy tags on video/audio latent rows."""

from __future__ import annotations

import os
import re
from typing import Any


def _caption_text(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for key in ("caption", "prompt", "text", "description", "instruction"):
        for source in (meta, row):
            val = source.get(key) if isinstance(source, dict) else None
            if isinstance(val, str) and val.strip():
                return val.strip()
    return str(row.get("caption", "") or row.get("text", "") or "")


def infer_modality(text: str) -> str:
    t = text.lower()
    speech = any(k in t for k in ("speech", "spoken", "dialogue", "voice", "lyrics", "read aloud", "vocal"))
    music = any(k in t for k in ("music", "guitar", "melody", "accompaniment", "song", "sung"))
    sound = any(k in t for k in ("bark", "noise", "sound", "environment", "rubbing", "canine", "dog"))
    flags = [speech, music, sound]
    if sum(flags) >= 2:
        parts = []
        if sound:
            parts.append("sound")
        if music:
            parts.append("music")
        if speech:
            parts.append("speech")
        return "-".join(parts)
    if speech:
        return "speech"
    if music:
        return "music"
    if sound:
        return "sound"
    return "mix"


def infer_complexity(text: str) -> str:
    t = text.lower()
    if re.search(r"multi[- ]?audio|audio1.*audio2|two audio", t):
        return "multi-audio"
    if "round" in t and ("first" in t or "second" in t or "third" in t):
        return "multi-round"
    if "hop" in t or ("first" in t and "second" in t and "author" in t):
        return "multi-hop"
    if ";" in text or " while " in t or " without " in t and " while " in t:
        return "multi-instruction"
    if re.search(r"\[\d+\.?\d*s:", text):
        return "multi-part"
    return "single"


def infer_granularity(text: str) -> list[str]:
    t = text.lower()
    if any(k in t for k in ("segment", "slice", "[", "0.", "extract", "remove", "bark")):
        return ["local"]
    return ["global"]


def _complexity_proxy(complexity: str) -> float:
    if complexity == "single":
        return 1.0
    if complexity in ("multi-part", "multi-instruction"):
        return 0.85
    if complexity in ("multi-hop", "multi-round"):
        return 0.75
    if complexity == "multi-audio":
        return 0.7
    return 0.8


def build_mmae_sidecar(row: dict[str, Any]) -> dict[str, Any]:
    existing = (row.get("meta") or {}).get("mmae") if isinstance(row.get("meta"), dict) else None
    if isinstance(existing, dict) and existing.get("sample_id"):
        return existing
    text = _caption_text(row)
    preview = text[:80] if text else ""
    complexity = infer_complexity(text)
    return {
        "enabled": True,
        "modality": infer_modality(text),
        "complexity": complexity,
        "granularity": infer_granularity(text),
        "edit_complexity_proxy": round(_complexity_proxy(complexity), 3),
        "rubric_count": int(existing.get("rubric_count", 0)) if isinstance(existing, dict) else 0,
        "instruction_preview": preview,
        "benchmark": "arXiv:2606.07229",
    }


def annotate_video_latent_data(row: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    """Tag latent rows with inferred MMAE taxonomy when ``GOPEX_MMAE_ENABLE=1``."""
    if os.environ.get("GOPEX_MMAE_ENABLE", "0") != "1":
        return row
    meta = dict(row.get("meta") or {})
    meta["mmae"] = build_mmae_sidecar(row)
    out = dict(row)
    out["meta"] = meta
    return out


def annotate_audio_save_data(data: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    """Mirror MMAE taxonomy on audio latent sidecars when ``GOPEX_MMAE_ENABLE=1``."""
    if os.environ.get("GOPEX_MMAE_ENABLE", "0") != "1":
        return data
    out = dict(data)
    meta = dict(out.get("meta") or {})
    meta["mmae"] = build_mmae_sidecar(out)
    out["meta"] = meta
    out["mmae"] = meta["mmae"]
    return out
