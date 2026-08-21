"""Optional AV-fold metadata hook for bioacoustic clip rows."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(row: dict[str, Any]) -> dict[str, Any]:
    """Tag rows with MyGardenBird provenance hints for downstream prep."""
    path = str(row.get("path") or row.get("audio_path") or "")
    is_xc = "xc" in path.lower() or str(row.get("source_id", "")).startswith("xc")
    return {
        **row,
        "mygardenbird_xc_hint": is_xc,
        "mygardenbird_clip_duration_s": 3.0,
    }
