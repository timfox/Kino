"""Framework cards, tables, and CPU evaluation demos."""

from __future__ import annotations

from typing import Any

from ltx_trainer.qwen3_tts.config import Qwen3TtsConfig


def framework_card(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    return {
        "title": c.title,
        "framework": c.framework,
        "license": c.license,
        "github": c.github,
        "pip_package": c.pip_package,
        "transformers_pin": c.transformers_pin,
        "tokenizer": "Qwen3-TTS-Tokenizer-12Hz",
        "architectures": ("12Hz Base", "12Hz CustomVoice", "12Hz VoiceDesign"),
        "params": ("0.6B", "1.7B"),
        "languages": list(c.languages),
        "speakers": list(c.speakers),
        "modes": list(c.generation_modes),
        "comfy_reference": c.comfy_reference,
    }


def speaker_table(cfg: Qwen3TtsConfig | None = None) -> list[dict[str, str]]:
    c = cfg or Qwen3TtsConfig()
    return [
        {"speaker": name, "description": c.speaker_descriptions.get(name, "")}
        for name in c.speakers
    ]


def model_catalog(cfg: Qwen3TtsConfig | None = None) -> list[dict[str, str]]:
    c = cfg or Qwen3TtsConfig()
    return [
        {"id": c.hf_base_17b, "role": "voice_clone", "size": "1.7B"},
        {"id": c.hf_base_06b, "role": "voice_clone", "size": "0.6B"},
        {"id": c.hf_voice_design, "role": "voice_design", "size": "1.7B"},
        {"id": c.hf_custom_17b, "role": "custom_voice", "size": "1.7B"},
        {"id": c.hf_custom_06b, "role": "custom_voice", "size": "0.6B"},
        {"id": c.hf_tokenizer, "role": "tokenizer", "size": "—"},
    ]


def evaluation_demo(*, seed: int = 42, cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    return {
        "seed": seed,
        "framework": framework_card(c),
        "speakers": speaker_table(c),
        "models": model_catalog(c),
        "generation_defaults": {
            "top_p": c.top_p,
            "top_k": c.top_k,
            "temperature": c.temperature,
            "repetition_penalty": c.repetition_penalty,
        },
    }


def benchmarks_bundle(cfg: Qwen3TtsConfig | None = None) -> dict[str, Any]:
    c = cfg or Qwen3TtsConfig()
    return {
        "speaker_table": speaker_table(c),
        "model_catalog": model_catalog(c),
        "languages": list(c.languages),
        "attention_mechanisms": list(c.attention_mechanisms),
    }


def headline_results() -> dict[str, Any]:
    return {
        "streaming_latency_ms": 97,
        "languages": 10,
        "modes": ("custom_voice", "voice_design", "voice_clone", "dialogue"),
    }
