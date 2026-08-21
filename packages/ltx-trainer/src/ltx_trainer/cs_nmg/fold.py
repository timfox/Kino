"""Optional AV-fold metadata hook for CS speech corpora."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(row: dict[str, Any]) -> dict[str, Any]:
    """Tag rows that may contain code-switching for downstream prep."""
    text = str(row.get("text") or row.get("caption") or "")
    has_latin = any(ch.isascii() and ch.isalpha() for ch in text)
    has_non_latin = any(ord(ch) > 127 for ch in text)
    return {
        **row,
        "cs_nmg_code_switch_hint": has_latin and has_non_latin,
    }
