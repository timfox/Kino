"""AV-fold sidecar for SEAM interview guardrail metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("seam", {}))
    sidecar.setdefault("framework", "SEAM")
    sidecar.setdefault("task", "scripted_vs_spontaneous")
    sidecar.setdefault("window_s", 8.0)
    sidecar.setdefault("shortcut_aware", True)
    data = dict(data)
    data["seam"] = sidecar
    return data
