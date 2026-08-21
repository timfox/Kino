"""Prediction and dataset WAV path resolution for MMAE eval."""

from __future__ import annotations

from pathlib import Path


def resolve_prediction_wav(predictions_dir: Path, sample_id: str) -> Path | None:
    """Accept ``{id}/output.wav``, ``{id}/audio_output.wav``, or ``{id}.wav``."""
    candidates = [
        predictions_dir / sample_id / "output.wav",
        predictions_dir / sample_id / "audio_output.wav",
        predictions_dir / f"{sample_id}.wav",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None
