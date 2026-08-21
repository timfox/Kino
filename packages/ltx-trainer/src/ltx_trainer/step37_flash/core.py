"""Core metrics and cost helpers."""

from __future__ import annotations

from typing import Any

from ltx_trainer.step37_flash.constants import (
    ACTIVE_PARAMS_PER_TOKEN_B,
    PEAK_THROUGHPUT_TPS,
    PRICING_PER_M,
    TOTAL_PARAMS_B,
)


def active_param_fraction() -> float:
    return ACTIVE_PARAMS_PER_TOKEN_B / TOTAL_PARAMS_B


def estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_hit_ratio: float = 0.0,
) -> dict[str, float]:
    """Rough USD cost from published $/M token rates."""
    cache_hit_ratio = max(0.0, min(1.0, cache_hit_ratio))
    hit = int(input_tokens * cache_hit_ratio)
    miss = input_tokens - hit
    cost = (
        miss * PRICING_PER_M["input_cache_miss"] / 1_000_000
        + hit * PRICING_PER_M["input_cache_hit"] / 1_000_000
        + output_tokens * PRICING_PER_M["output"] / 1_000_000
    )
    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "cache_hit_ratio": cache_hit_ratio,
        "usd": round(cost, 6),
    }


def throughput_card() -> dict[str, Any]:
    return {
        "peak_tokens_per_second": PEAK_THROUGHPUT_TPS,
        "active_params_b": ACTIVE_PARAMS_PER_TOKEN_B,
        "total_params_b": TOTAL_PARAMS_B,
        "sparsity_note": "~11B active of 198B total MoE VLM",
    }
