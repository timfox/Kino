"""FGNV encoding smoke."""

from __future__ import annotations

import math
from typing import Any


def circumplex_embedding(*, arousal: float, valence: float) -> list[float]:
    """Russell circumplex features (arousal, valence, radius, angle)."""
    theta = math.atan2(arousal, valence)
    radius = math.sqrt(arousal * arousal + valence * valence)
    return [arousal, valence, radius, theta]


def predict_emotion_from_cues(*, arousal: float, valence: float) -> str:
    if valence < -0.3:
        return "sad"
    if valence > 0.3:
        return "happy"
    if arousal > 0.5:
        return "excited"
    return "neutral"


def demo_recognition(text: str, *, arousal: float, valence: float) -> dict[str, Any]:
    from ltx_trainer.fgnv.parsers import encode_utterance

    enc = encode_utterance(text)
    label = predict_emotion_from_cues(arousal=arousal, valence=valence)
    return {
        "predicted_emotion": label,
        "encoding": enc,
        "embedding": circumplex_embedding(arousal=arousal, valence=valence),
    }


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.fgnv.parsers import encode_utterance

    enc = encode_utterance("<(crying) wuuuuu whep> what did you do")
    return {
        "has_nv": enc["has_nv"],
        "style": enc["style"],
        "utterance_len": len(enc["token_id"]),
    }
