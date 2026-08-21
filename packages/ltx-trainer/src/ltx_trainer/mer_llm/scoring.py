"""Toy scoring helpers for survey demos (no real MLLM)."""

from __future__ import annotations

from typing import Any


def paradigm_lift(small_sota: float, mllm_zero_shot: float, mer_with_llms: float) -> dict[str, float]:
    """Gains vs baselines (Fig. 1b style)."""
    return {
        "vs_small_sota": round(mer_with_llms - small_sota, 2),
        "vs_mllm_zero_shot": round(mer_with_llms - mllm_zero_shot, 2),
    }


def explainable_mer_stub(
    *,
    observation: str,
    emotion: str,
    confidence: float = 0.85,
) -> dict[str, Any]:
    """Minimal explainable MER output (Fig. 6a style)."""
    conf = max(0.0, min(1.0, confidence))
    return {
        "thinking": f"<thinking>{observation}</thinking>",
        "answer": f"<answer>{emotion}</answer>",
        "confidence_verbalization": (
            f"I am {conf:.0%} confident in this assessment." if conf >= 0.5 else "uncertain"
        ),
    }
