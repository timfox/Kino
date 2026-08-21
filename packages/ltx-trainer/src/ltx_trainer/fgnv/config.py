"""Fine-grained non-verbal emotional TTS (arXiv:2605.25504)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FgnvConfig:
    paper_arxiv: str = "arXiv:2605.25504"
    demo_page: str = "https://37integer.github.io/FINE-GRAINED-NON-VERBAL-TTS/"
    sample_rate_hz: int = 22050
    mel_dim: int = 80
    backbone: str = "Grad-TTS"
    vocoder: str = "HiFi-GAN"
    training_iterations: int = 400_000
    mixed_verbal_hours: float = 9.0
    nv_utterances: int = 739
    nv_source_recordings: int = 360
    female_speakers: int = 60
    eval_participants: int = 15
    eval_sentences: int = 20
    emotions: tuple[str, ...] = ("happy", "sad", "fear", "anger")
    silence_threshold_dbfs: float = -40.0
    min_silence_ms: int = 200
    silence_buffer_ms: int = 100
