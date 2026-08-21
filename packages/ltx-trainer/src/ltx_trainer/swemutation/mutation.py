"""Agentic mutation framework and strategy taxonomy (Sec. 3.1, Appendix A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.swemutation.config import MUTATION_STRATEGY_GROUPS, MutationStrategy


@dataclass(frozen=True)
class JudgeVerdict:
    """Judge module constraints (Sec. 3.1.4)."""

    in_golden_files: bool
    compiles: bool
    fails_f2p: bool

    @property
    def valid(self) -> bool:
        return self.in_golden_files and self.compiles and self.fails_f2p


def judge_mutant(
    *,
    modified_only_golden_files: bool,
    compiles: bool,
    fails_at_least_one_f2p: bool,
) -> JudgeVerdict:
    return JudgeVerdict(
        in_golden_files=modified_only_golden_files,
        compiles=compiles,
        fails_f2p=fails_at_least_one_f2p,
    )


def self_play_select(
    survival_counts: list[int],
    *,
    threshold: int = 3,
    top_fraction: float = 0.5,
) -> list[int]:
    """Keep top 50% mutants that evaded > threshold generated test suites (Sec. 3.1.5)."""
    eligible = [i for i, c in enumerate(survival_counts) if c > threshold]
    if not eligible:
        return []
    ranked = sorted(eligible, key=lambda i: survival_counts[i], reverse=True)
    k = max(1, int(len(ranked) * top_fraction))
    return ranked[:k]


STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "A_api_contracts": "API specs: defaults, arg order, exception types, read-only violations",
    "B_boundaries": "Boundaries: off-by-one, null guards, boolean inversion",
    "C_type_shape": "Types: coercion, precision, bytes vs text",
    "D_stateful": "State: init/reset, idempotency, sequencing, globals",
    "E_test_alignment": "Assertions: error messages, implicit→explicit params",
}


def mutation_strategy_catalog() -> list[dict[str, Any]]:
    return [
        {"id": gid, "description": STRATEGY_DESCRIPTIONS[gid]}
        for gid in MUTATION_STRATEGY_GROUPS
    ]


def compare_mutation_methods() -> dict[str, dict[str, str]]:
    """Table 1 benchmark characteristics vs prior work."""
    return {
        "rule_based": {"mutation": "cosmic-ray operators", "agentic_loop": "no"},
        "few_shot": {"mutation": "LLM few-shot from strategy pool", "agentic_loop": "no"},
        "agentic": {"mutation": "Locate + Mutation + Judge + Self-Play", "agentic_loop": "yes"},
    }
