"""GOPEX / LTX integration — BES for caption and agent sample generation."""

from __future__ import annotations

import os
from typing import Any

from ltx_trainer.bes.config import BESConfig
from ltx_trainer.bes.core import post_training_sample_plan


def ltx_sample_generation_plan(
    *,
    backend: str = "vllm",
    use_case: str = "agent_rollouts",
) -> dict[str, Any]:
    """
    Plan for using BES instead of best-of-N in GOPEX training loops.

    Set ``GOPEX_AGENT_SEARCH=bes`` (proposed) to route ``dataset_split_and_caption``
    and parallel-prep reasoning through BES forward+backward search.
    """
    cfg = BESConfig()
    base = post_training_sample_plan()
    return {
        **base,
        "env": {
            "GOPEX_AGENT_SEARCH": "bes",
            "GOPEX_BES_BUDGET": str(cfg.budget_calls),
            "GOPEX_BES_DECOMPOSE_EVERY": str(cfg.decompose_interval),
            "GOPEX_BES_ALPHA": str(cfg.alpha),
        },
        "policy_backend": backend,
        "use_case": use_case,
        "verifier_stack": [
            "rule_based (K&K, code exec)",
            "embedding similarity (MuSiQue sub-goals)",
            "LLM judge + bucket rank (open programs)",
        ],
        "current_default": os.environ.get("GOPEX_AGENT_SEARCH", "best_of_n"),
    }


def gopex_reasoning_hook() -> dict[str, Any]:
    """Bridge to gopex_tot / vLLM messenger for hard reasoning prompts."""
    return {
        "complementary": "gopex_tot (ToT beam/BFS) vs bes (evolution + backward goals)",
        "when_bes": "sparse terminal reward, need to recombine partial trajectories",
        "when_tot": "step-wise heuristic scores without explicit sub-goal tree",
        "docs": "documents/BES.md",
    }
