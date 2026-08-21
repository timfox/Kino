"""Terminal reward S(y) for PRM supervision (§2.3, Appendix C)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.constants import REWARD_WEIGHTS


def terminal_reward(
    *,
    compiled: bool,
    executed: bool,
    validators: dict[str, float] | None = None,
    validator_weights: dict[str, float] | None = None,
) -> float:
    """
    S(y) = 0.30·C + 0.25·E + 0.45 · (weighted mean of available validators).

    Benchmark metric uses executable validation only; mixed reward is for PRM training.
    """
    c = 1.0 if compiled else 0.0
    e = 1.0 if executed else 0.0
    base = REWARD_WEIGHTS["compile"] * c + REWARD_WEIGHTS["execute"] * e
    if not validators:
        return base
    weights = validator_weights or {k: 1.0 for k in validators}
    avail = {k: v for k, v in validators.items() if k in weights}
    if not avail:
        return base
    w_sum = sum(weights[k] for k in avail)
    v_mean = sum(weights[k] * avail[k] for k in avail) / w_sum
    return base + REWARD_WEIGHTS["validators"] * v_mean


def reward_card() -> dict[str, Any]:
    return {
        "compile_weight": REWARD_WEIGHTS["compile"],
        "execute_weight": REWARD_WEIGHTS["execute"],
        "validator_pool_weight": REWARD_WEIGHTS["validators"],
        "validators": [
            "pass/fail integrated validation",
            "checksum / output matching",
            "benchmark metrics",
            "Jina code-embedding similarity",
            "GPT-5-mini judge",
            "variable-comparison heuristic",
        ],
        "benchmark_metric": "executable validation (compile + run + validate)",
        "common_effective_weights": {
            "validation": 0.2423,
            "jina_similarity": 0.1038,
            "gpt5_mini_judge": 0.1038,
        },
    }
