"""Evaluation Table I and scenario metadata (Section V)."""

from __future__ import annotations

from typing import Any

TABLE1_EVALUATION = [
    {"scenario": "Knowledge Query", "cases": 10, "repetitions": 1, "successful": 9},
    {"scenario": "SAT Planning", "cases": 4, "repetitions": 5, "successful": 4},
    {"scenario": "UNSAT Planning", "cases": 4, "repetitions": 5, "successful": 3},
    {"scenario": "Adaptive Planning", "cases": 5, "repetitions": 5, "successful": 5},
]

RESEARCH_QUESTIONS = {
    "RQ1": "Natural-language interaction for capability knowledge and planning formulation",
    "RQ2": "Understandable explanation of SAT and UNSAT planning results",
    "RQ3": "Actionable adaptation from UNSAT feedback and runtime failure reports",
}

HITL_CHECKPOINTS = [
    "confirm_required_capability_before_planning",
    "approve_capability_model_adaptation_after_unsat",
    "confirm_capability_update_on_runtime_failure",
]

UNSAT_EXAMPLE_CONFLICTS = [
    {
        "id": "conflict_depth",
        "text": "Required output depth 2 mm outside drill acceptable range 5–10 mm",
        "repair": "Adjust required depth to 5 mm",
    },
    {
        "id": "conflict_station",
        "text": "Required station ID 15 does not match provided station ID 3",
        "repair": "Align required station ID to 3",
    },
]


def benchmarks_bundle() -> dict[str, Any]:
    total_cases = sum(r["cases"] for r in TABLE1_EVALUATION)
    total_success = sum(r["successful"] for r in TABLE1_EVALUATION)
    return {
        "table1": TABLE1_EVALUATION,
        "total_test_cases": total_cases,
        "total_successful_categories": total_success,
        "research_questions": RESEARCH_QUESTIONS,
        "hitl_checkpoints": HITL_CHECKPOINTS,
        "agents": [
            "router",
            "knowledge_retrieval",
            "capability_mapper",
            "planning",
            "analyze_adaptation",
            "repair",
        ],
        "components": [
            "capability_grounding",
            "symbolic_planning",
            "result_interpretation",
            "planning_adaptation",
        ],
        "unsat_example": UNSAT_EXAMPLE_CONFLICTS,
        "mps500_repo": "https://github.com/hsu-aut/MPS500-Capabilities",
    }
