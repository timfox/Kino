"""27 fine-grained MMAR categories and routing strategies (Table 3)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class RoutingStrategy(str, Enum):
    LLM_REASONING = "llm_reasoning_and_selection"
    VLM_SPECTRAL = "vlm_empowered_spectral_reasoning"
    DIRECT_QWEN = "direct_selection_qwen3_omni_thinking"
    DIRECT_STEP = "direct_selection_step_audio_r1"


CATEGORY_REGISTRY: list[dict[str, Any]] = [
    # LLM Reasoning And Selection
    {"id": "semantic_logic", "name": "Semantic Logic Reasoning", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "emotion_intention", "name": "Emotion and Intention Inference", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "music_theory", "name": "Music Theory Reasoning", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "style_genre", "name": "Style and Genre Classification", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "language_culture", "name": "Language/Culture Recognition", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "multi_speaker", "name": "Multi-speaker Reasoning", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "music_structure", "name": "Music Structure Understanding", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "aesthetic_quality", "name": "Aesthetic and Quality Judgment", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "temporal_order", "name": "Temporal Order Reasoning", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "pattern_change", "name": "Pattern Change Detection", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "loudness_intensity", "name": "Loudness and Intensity Detection", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "multi_source", "name": "Multi-source Reasoning", "strategy": RoutingStrategy.LLM_REASONING},
    {"id": "noise_anomaly", "name": "Noise and Anomaly Detection", "strategy": RoutingStrategy.LLM_REASONING},
    # VLM-Empowered Spectral Reasoning
    {"id": "event_counting", "name": "Event Counting and Statistics", "strategy": RoutingStrategy.VLM_SPECTRAL},
    {"id": "rhythm_beat", "name": "Rhythm and Beat Analysis", "strategy": RoutingStrategy.VLM_SPECTRAL},
    {"id": "instrument_id", "name": "Instrument Identification", "strategy": RoutingStrategy.VLM_SPECTRAL},
    {"id": "distance_perspective", "name": "Distance and Perspective Est.", "strategy": RoutingStrategy.VLM_SPECTRAL},
    {"id": "audio_difference", "name": "Audio Difference Comparison", "strategy": RoutingStrategy.VLM_SPECTRAL},
    {"id": "pitch_freq", "name": "Pitch and Freq. Identification", "strategy": RoutingStrategy.VLM_SPECTRAL},
    # Direct Selection — Qwen3-Omni-Thinking
    {"id": "env_scene", "name": "Environmental Scene Recognition", "strategy": RoutingStrategy.DIRECT_QWEN},
    {"id": "speech_content", "name": "Speech Content Recognition", "strategy": RoutingStrategy.DIRECT_QWEN},
    {"id": "speaker_identity", "name": "Speaker Identity Analysis", "strategy": RoutingStrategy.DIRECT_QWEN},
    {"id": "timbre_texture", "name": "Timbre and Texture Recognition", "strategy": RoutingStrategy.DIRECT_QWEN},
    {"id": "spatial_localization", "name": "Spatial Localization", "strategy": RoutingStrategy.DIRECT_QWEN},
    # Direct Selection — Step-Audio-R1
    {"id": "sound_source", "name": "Sound Source Identification", "strategy": RoutingStrategy.DIRECT_STEP},
    {"id": "acoustic_quality", "name": "Acoustic Quality Assessment", "strategy": RoutingStrategy.DIRECT_STEP},
    {"id": "duration_estimation", "name": "Duration Estimation", "strategy": RoutingStrategy.DIRECT_STEP},
]


def category_registry() -> list[dict[str, Any]]:
    return [dict(c) for c in CATEGORY_REGISTRY]


def routing_strategy_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in CATEGORY_REGISTRY:
        key = row["strategy"].value
        counts[key] = counts.get(key, 0) + 1
    return counts


def lookup_category(category_id: str) -> dict[str, Any]:
    for row in CATEGORY_REGISTRY:
        if row["id"] == category_id:
            return dict(row)
    raise KeyError(f"unknown category: {category_id}")
