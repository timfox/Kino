"""Configuration for SWE-Mutation (Sun et al., arXiv:2605.22175)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TaskName = Literal["test_generation", "test_repair"]
MutationStrategy = Literal["rule_based", "few_shot", "agentic"]
AgentFramework = Literal["mini_swe_agent", "claude_code"]

MUTATION_STRATEGY_GROUPS: tuple[str, ...] = (
    "A_api_contracts",
    "B_boundaries",
    "C_type_shape",
    "D_stateful",
    "E_test_alignment",
)

MULTILINGUAL_LANGS: tuple[str, ...] = (
    "python",
    "c",
    "cpp",
    "java",
    "typescript",
    "javascript",
    "rust",
    "go",
    "php",
    "ruby",
)


@dataclass
class SWEMutationConfig:
    """Defaults from paper Sec. 3–4."""

    python_instances: int = 500
    multilingual_instances: int = 300
    total_mutants: int = 2636
    python_mutants: int = 1664
    multilingual_mutants: int = 972
    mutants_per_instance_min: int = 3
    mutants_per_instance_max: int = 5
    self_play_candidates: int = 10  # N diverse candidates
    self_play_survival_threshold: int = 3  # evade >3 test suites
    anchor_tau: float = 0.2  # not PIU - ignore
    # Table 2 DeepSeek-V3.1 test repair (Claude Code)
    reference_vrr_repair: float = 58.20
    reference_rdr_repair: float = 68.36
    # Table 3 DeepSeek-V3.1 test generation (Mini-Swe-Agent)
    reference_vrr_gen: float = 10.20
    reference_rdr_gen: float = 36.15
    # Table 5 agentic vs rule-based RDR drop
    rule_based_rdr_avg: float = 71.04
    agentic_rdr_avg: float = 39.81
