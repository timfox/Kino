"""AV-fold sidecar metadata for VISA routing hints."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    """Attach VISA agent-track metadata to precomputed audio shards."""
    sidecar = dict(data.get("visa", {}))
    sidecar.setdefault("agent_track", "interspeech_2026_arc")
    sidecar.setdefault("routing_categories", 27)
    sidecar.setdefault("paradigm", "lalm_as_tool")
    data = dict(data)
    data["visa"] = sidecar
    return data
