"""Qwen3-TTS — Alibaba Qwen speech synthesis integration."""

from ltx_trainer.qwen3_tts.attention import detect_best_attention, resolve_attention
from ltx_trainer.qwen3_tts.config import GenerationMode, ModelChoice, Qwen3TtsConfig
from ltx_trainer.qwen3_tts.dialogue import DialogueLine, RoleBank, parse_dialogue_script, validate_script_roles
from ltx_trainer.qwen3_tts.eval import eval_smoke, pipeline_demo
from ltx_trainer.qwen3_tts.mock import evaluation_smoke
from ltx_trainer.qwen3_tts.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    model_catalog,
    speaker_table,
)
from ltx_trainer.qwen3_tts.runtime import GenerationParams, Qwen3TtsRuntime, write_wav
from ltx_trainer.qwen3_tts.upstream import (
    build_infer_argv,
    check_transformers_version,
    doctor,
    install_plan,
    models_root,
    package_status,
    resolve_model_path,
    upstream_knowledge,
    voices_dir,
)
from ltx_trainer.qwen3_tts.voice_bank import SavedVoice, list_saved_voices, load_voice, save_voice

__all__ = [
    "GenerationMode",
    "ModelChoice",
    "Qwen3TtsConfig",
    "Qwen3TtsRuntime",
    "GenerationParams",
    "DialogueLine",
    "RoleBank",
    "SavedVoice",
    "benchmarks_bundle",
    "build_infer_argv",
    "check_transformers_version",
    "detect_best_attention",
    "doctor",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "install_plan",
    "list_saved_voices",
    "load_voice",
    "model_catalog",
    "models_root",
    "package_status",
    "parse_dialogue_script",
    "pipeline_demo",
    "resolve_attention",
    "resolve_model_path",
    "save_voice",
    "speaker_table",
    "upstream_knowledge",
    "validate_script_roles",
    "voices_dir",
    "write_wav",
]
