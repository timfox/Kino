"""Hybrid capability-based SMT planning assistance (arXiv:2605.28666)."""

from __future__ import annotations

from dataclasses import dataclass, field

INTENT_TYPES = ("knowledge_query", "planning_request", "runtime_failure")
AGENT_ROLES = (
    "router",
    "knowledge_retrieval",
    "capability_mapper",
    "planning",
    "analyze_adaptation",
    "repair",
)
COMPONENTS = (
    "capability_grounding",
    "symbolic_planning",
    "result_interpretation",
    "planning_adaptation",
)
SCENARIO_TYPES = ("knowledge_query", "sat_planning", "unsat_planning", "adaptive_planning")


@dataclass
class CbpAssistConfig:
    arxiv: str = "2605.28666"
    mps500_repo: str = "https://github.com/hsu-aut/MPS500-Capabilities"
    institution: str = "Helmut Schmidt University (HSU)"
    llm_default: str = "gpt-4o-mini"
    llm_analyzer: str = "gpt-4o"
    orchestration: str = "LangGraph"
    test_cases_total: int = 23
    evaluation_repetitions: dict[str, int] = field(
        default_factory=lambda: {
            "knowledge_query": 1,
            "sat_planning": 5,
            "unsat_planning": 5,
            "adaptive_planning": 5,
        }
    )
    table1_successful: dict[str, int] = field(
        default_factory=lambda: {
            "knowledge_query": 9,
            "sat_planning": 4,
            "unsat_planning": 3,
            "adaptive_planning": 5,
        }
    )
    table1_cases: dict[str, int] = field(
        default_factory=lambda: {
            "knowledge_query": 10,
            "sat_planning": 4,
            "unsat_planning": 4,
            "adaptive_planning": 5,
        }
    )
