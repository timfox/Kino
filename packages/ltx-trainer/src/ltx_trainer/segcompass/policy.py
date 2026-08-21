"""MLLM policy / MCoT rollout stub (Sec. 3.1, Eq. 1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RolloutSpec:
    num_concentration_tokens: int
    cot_token_estimate: int
    includes_ref_token: bool


def mcot_rollout_spec(
    instruction: str,
    *,
    num_slots: int = 4,
) -> RolloutSpec:
    """Estimate CoT + Ks concentration tokens from instruction complexity."""
    words = len(instruction.split())
    cot_tokens = max(16, min(512, words * 8))
    return RolloutSpec(
        num_concentration_tokens=num_slots,
        cot_token_estimate=cot_tokens,
        includes_ref_token=True,
    )


def policy_logprob_ratio(current: float, old: float) -> float:
    """π_θ / π_θ_old for GRPO ratio term."""
    if old <= 0:
        return 0.0
    return current / old
