"""MoE VLM architecture summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.step37_flash.constants import (
    ACTIVE_PARAMS_PER_TOKEN_B,
    LANGUAGE_PARAMS_B,
    MAX_CONTEXT_TOKENS,
    PEAK_THROUGHPUT_TPS,
    REASONING_LEVELS,
    TOTAL_PARAMS_B,
    VISION_PARAMS_B,
)


def architecture_card() -> dict[str, Any]:
    active_ratio = ACTIVE_PARAMS_PER_TOKEN_B / TOTAL_PARAMS_B
    return {
        "family": "sparse_mixture_of_experts_vlm",
        "total_params_b": TOTAL_PARAMS_B,
        "language_backbone_params_b": LANGUAGE_PARAMS_B,
        "vision_encoder_params_b": VISION_PARAMS_B,
        "active_params_per_token_b": ACTIVE_PARAMS_PER_TOKEN_B,
        "active_ratio": round(active_ratio, 4),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "peak_throughput_tokens_per_sec": PEAK_THROUGHPUT_TPS,
        "reasoning_levels": list(REASONING_LEVELS),
        "native_multimodal": True,
        "use_cases": [
            "financial report parsing at 256k",
            "multi-step search with verification",
            "concurrent coding agents",
            "UI/GUI → structured code",
        ],
    }


def reasoning_level_tradeoff(level: str) -> dict[str, str]:
    """Map reasoning level to speed/cost/depth tradeoff (qualitative)."""
    level = level.lower()
    table = {
        "low": {
            "speed": "fastest",
            "cost": "lowest",
            "depth": "minimal chain-of-thought",
        },
        "medium": {
            "speed": "balanced",
            "cost": "moderate",
            "depth": "default production",
        },
        "high": {
            "speed": "slowest",
            "cost": "highest",
            "depth": "maximum cognitive depth",
        },
    }
    if level not in table:
        raise ValueError(f"reasoning level must be one of {REASONING_LEVELS}, got {level!r}")
    return {"level": level, **table[level]}
