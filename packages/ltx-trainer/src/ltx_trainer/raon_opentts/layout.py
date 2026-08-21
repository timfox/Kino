"""Limitations note for Raon-OpenTTS stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "English-only pool and models in the paper; multilingual extension is future work.",
    "Stub omits full YouTube-Commons preprocessing (UVR, pyannote, Silero VAD, Whisper ASR) and distributed B200 training recipes.",
    "Benchmark numbers are excerpted from the paper; reproduce with official checkpoints and eval scripts from the Krafton repo.",
)
