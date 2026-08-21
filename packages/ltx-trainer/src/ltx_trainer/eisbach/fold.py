"""AV-fold sidecar for Eisbach music diffusion metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("eisbach", {}))
    sidecar.setdefault("framework", "Eisbach")
    sidecar.setdefault("barrier_lambda", 0.5)
    sidecar.setdefault("task", "structural_music_prior")
    data = dict(data)
    data["eisbach"] = sidecar
    return data
