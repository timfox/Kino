"""Benchmark excerpts (Agentboard, BFCL, search-agent latency)."""

from __future__ import annotations

from typing import Any


def agentboard_excerpt() -> list[dict[str, str]]:
    return [
        {"backbone": "LLaDA", "paradigm": "embodied", "note": "Repeated subgoal attempts"},
        {"backbone": "Dream", "paradigm": "embodied", "note": "Weak replan under feedback"},
        {"backbone": "AR baseline", "paradigm": "embodied", "note": "Reference branching behavior"},
    ]


def bfcl_excerpt() -> list[dict[str, str]]:
    return [
        {"metric": "schema_validity", "dllm": "degraded under mask noise", "ar": "higher"},
        {"metric": "executable_accuracy", "dllm": "tool arg precision drops", "ar": "reference"},
    ]


def search_agent_latency() -> dict[str, Any]:
    return {
        "paradigm": "P-ReAct",
        "gain_pct": 15,
        "comparable_accuracy": True,
        "note": "DLLM-Searcher vs mainstream LLM search agents",
    }


def paper_anchors() -> dict[str, str]:
    return {
        "bitter_lesson": "2601.12979",
        "dllm_agent": "2602.07451",
        "dllm_searcher": "2602.07035",
        "dlmasr_decode": "2605.29613",
        "constraint_tax": "2605.26128",
        "tot_search": "2605.28566",
    }
