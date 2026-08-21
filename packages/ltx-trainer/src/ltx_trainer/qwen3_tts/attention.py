"""Attention mechanism selection with auto-detect and graceful fallback."""

from __future__ import annotations

import importlib.util
from typing import Literal

from ltx_trainer.qwen3_tts.config import AttentionChoice

AttnImpl = Literal["flash_attention_2", "sdpa", "eager"]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_best_attention() -> AttentionChoice:
    """Mirror ComfyUI-Qwen-TTS priority: sage → flash → sdpa → eager."""
    if _has_module("sageattention"):
        return "sage_attn"
    if _has_module("flash_attn"):
        return "flash_attn"
    return "sdpa"


def resolve_attention(choice: AttentionChoice = "auto") -> tuple[AttentionChoice, AttnImpl, str | None]:
    """
    Return (requested_or_resolved, hf_attn_implementation, warning).

    ``qwen-tts`` accepts HuggingFace ``attn_implementation`` on ``from_pretrained``.
    SAGE is optional and may require upstream extras; we fall back when unavailable.
    """
    requested: AttentionChoice = detect_best_attention() if choice == "auto" else choice
    warning: str | None = None

    if requested == "sage_attn":
        if not _has_module("sageattention"):
            warning = "sage_attn unavailable; falling back to sdpa"
            return "sdpa", "sdpa", warning
        # Official qwen-tts wheel uses HF attn flags; sage may be injected by extras.
        return "sage_attn", "sdpa", None

    if requested == "flash_attn":
        if not _has_module("flash_attn"):
            warning = "flash_attn unavailable; falling back to sdpa"
            return "sdpa", "sdpa", warning
        return "flash_attn", "flash_attention_2", None

    if requested == "sdpa":
        return "sdpa", "sdpa", None

    return "eager", "eager", None
