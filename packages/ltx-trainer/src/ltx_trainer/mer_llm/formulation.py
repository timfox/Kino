"""MER-with-LLMs problem formulation (Sec. 2, Eq. 1)."""

from __future__ import annotations

from typing import Any


def autoregressive_mer_step(
    *,
    vocab_size: int = 32000,
    context_hash: int = 0,
    step: int = 0,
) -> dict[str, Any]:
    """Toy next-token choice: ri = argmax_r πθ(r | X, Q, R_<i) (deterministic stub)."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    token_id = abs(hash((context_hash, step))) % vocab_size
    return {
        "step": step,
        "token_id": token_id,
        "equation": "ri = argmax_r πθ(r | X, Q, R_<i)",
    }


def mer_with_llms_response_schema() -> dict[str, Any]:
    """Typical outputs under the paradigm: labels and/or explanations."""
    return {
        "closed_set": "emotion category from predefined label set",
        "open_vocabulary": "free-form emotion phrases (OV-MERD, Agent-MER)",
        "explainable": "chain-of-thought or natural-language rationale before <answer>",
        "confidence": "verbalized confidence (EmoCaliber)",
    }
