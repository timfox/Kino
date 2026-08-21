"""AV-fold sidecar for dots.tts TTS metadata."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    sidecar = dict(data.get("dotstts", {}))
    sidecar.setdefault("framework", "dots.tts")
    sidecar.setdefault("task", "continuous_ar_tts")
    sidecar.setdefault("latent_fps", 25.0)
    sidecar.setdefault("semantic_fps", 6.25)
    sidecar.setdefault("checkpoints", ["Pretrain", "SOAR", "MF"])
    data = dict(data)
    data["dotstts"] = sidecar
    return data
