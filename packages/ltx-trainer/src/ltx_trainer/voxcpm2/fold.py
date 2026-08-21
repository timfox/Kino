"""Optional AV-fold metadata hook for TTS clip rows."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or row.get("caption") or "")
    has_paren_desc = text.startswith("(") and ")" in text[:120]
    return {
        **row,
        "voxcpm2_voice_design_hint": has_paren_desc,
        "voxcpm2_output_hz": 48000,
    }
