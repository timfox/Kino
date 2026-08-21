"""AV-fold sidecar for AudioIM V2A style metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("audioim", {}))
    sidecar.setdefault("framework", "AudioIM")
    sidecar.setdefault("prompt_seconds", 3.0)
    sidecar.setdefault("style_factors", ["timbre", "tempo"])
    data = dict(data)
    data["audioim"] = sidecar
    return data
