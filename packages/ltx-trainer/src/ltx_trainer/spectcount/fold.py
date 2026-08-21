"""AV-fold sidecar for SpectCount LALM metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("spectcount", {}))
    sidecar.setdefault("framework", "SpectCount")
    sidecar.setdefault("synthetic_on_the_fly", True)
    sidecar.setdefault("objective", "pulse_counting")
    data = dict(data)
    data["spectcount"] = sidecar
    return data
