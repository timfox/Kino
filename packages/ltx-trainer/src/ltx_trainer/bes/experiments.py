"""Experiment presets from Appendix D."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bes.config import (
    BESConfig,
    KnightsKnavesPreset,
    MuSiQuePreset,
    OpenProblemPreset,
)


def knights_knaves_config() -> BESConfig:
    p = KnightsKnavesPreset()
    return BESConfig(
        budget_calls=p.search_budget,
        decompose_interval=p.decompose_interval,
        tau_start=p.tau_start,
        tau_end=p.tau_end,
        alpha=p.alpha,
        group_size=p.group_size,
    )


def musique_config() -> BESConfig:
    p = MuSiQuePreset()
    return BESConfig(
        budget_calls=p.search_budget,
        tau_start=p.tau_start,
        tau_end=p.tau_end,
        alpha=p.alpha,
        group_size=p.group_size,
    )


def open_problem_card() -> dict[str, Any]:
    p = OpenProblemPreset()
    return {
        "base_framework": "ShinkaEvolve",
        "backbone": "gpt-5",
        "reasoning_effort": "high",
        "num_generations": p.num_generations,
        "stagnation_trigger": p.stagnation_generations,
        "backward_alpha": p.alpha,
        "api_budget_usd": p.api_budget_usd,
        "bes_operators": ["combine", "deletion", "translocation", "crossover"],
        "operator_realization": "LLM prompts (programs are not token-concatenable)",
    }


def experiments_bundle() -> dict[str, Any]:
    return {
        "logical_reasoning": {
            "dataset": "Knights-and-Knaves",
            "base_model": "Gemma-3-1B-it",
            "baselines": ["GRPO", "MaxRL"],
            "config": knights_knaves_config().__dict__,
        },
        "multi_hop": {
            "dataset": "MuSiQue 3-4 hop",
            "base_models": ["Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct"],
            "baselines": ["GRPO", "Tree-GRPO"],
            "config": musique_config().__dict__,
        },
        "open_problems": open_problem_card(),
    }
