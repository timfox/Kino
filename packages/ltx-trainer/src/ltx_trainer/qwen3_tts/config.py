"""Qwen3-TTS — Alibaba Qwen speech synthesis (voice clone, design, custom)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GenerationMode = Literal[
    "custom_voice",
    "voice_design",
    "voice_clone",
    "voice_clone_prompt",
    "dialogue",
]

ModelChoice = Literal["0.6B", "1.7B"]
AttentionChoice = Literal["auto", "sage_attn", "flash_attn", "sdpa", "eager"]


@dataclass
class Qwen3TtsConfig:
    title: str = "Qwen3-TTS"
    framework: str = "Qwen3-TTS"
    license: str = "Apache-2.0"
    github: str = "https://github.com/Qwen/Qwen3-TTS"
    comfy_reference: str = "https://github.com/flybirdxx/ComfyUI-Qwen-TTS"
    pip_package: str = "qwen-tts"
    transformers_pin: str = "4.57.3"
    accelerate_pin: str = "1.12.0"

    # Hub model ids (12 Hz family)
    hf_tokenizer: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    hf_base_17b: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    hf_base_06b: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    hf_voice_design: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    hf_custom_17b: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    hf_custom_06b: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    # Inference defaults (ComfyUI-Qwen-TTS parity)
    dtype: str = "bfloat16"
    default_language: str = "Auto"
    top_p: float = 0.8
    top_k: int = 20
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    dialogue_pause_seconds: float = 0.35
    dialogue_batch_size: int = 4
    max_roles: int = 8

    languages: tuple[str, ...] = (
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    )

    speakers: tuple[str, ...] = (
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    )

    speaker_descriptions: dict[str, str] = field(
        default_factory=lambda: {
            "Vivian": "Bright, slightly edgy young female voice (Chinese native).",
            "Serena": "Warm, gentle young female voice (Chinese native).",
            "Uncle_Fu": "Seasoned male voice with a low, mellow timbre (Chinese native).",
            "Dylan": "Youthful Beijing male voice with a clear, natural timbre.",
            "Eric": "Lively Chengdu male voice with a slightly husky brightness.",
            "Ryan": "Dynamic male voice with strong rhythmic drive (English native).",
            "Aiden": "Sunny American male voice with a clear midrange (English native).",
            "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre.",
            "Sohee": "Warm Korean female voice with rich emotion.",
        }
    )

    generation_modes: tuple[GenerationMode, ...] = (
        "custom_voice",
        "voice_design",
        "voice_clone",
        "voice_clone_prompt",
        "dialogue",
    )

    attention_mechanisms: tuple[AttentionChoice, ...] = (
        "auto",
        "sage_attn",
        "flash_attn",
        "sdpa",
        "eager",
    )

    def model_id_for(self, mode: GenerationMode, choice: ModelChoice = "1.7B") -> str:
        if mode == "voice_design":
            return self.hf_voice_design
        if mode == "custom_voice":
            return self.hf_custom_17b if choice == "1.7B" else self.hf_custom_06b
        # clone / prompt / dialogue use Base checkpoints
        return self.hf_base_17b if choice == "1.7B" else self.hf_base_06b
