"""AV-fold sidecar for CogAudio-LLM affective metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("cogaudio_llm", {}))
    sidecar.setdefault("framework", "CogAudio-LLM")
    sidecar.setdefault("eips_steps", 4)
    sidecar.setdefault("semantic_decoupling", "LIME-440K")
    data = dict(data)
    data["cogaudio_llm"] = sidecar
    return data
