"""Taxonomy: DiffuAgent roles, P-ReAct, deployment guidance."""

from __future__ import annotations

from typing import Any


def bitter_lesson_findings() -> list[dict[str, str]]:
    return [
        {
            "setting": "Embodied (Agentboard)",
            "finding": "dLLM backbones repeat attempts; weak branching under temporal feedback",
        },
        {
            "setting": "Tool-calling (BFCL)",
            "finding": "Diffusion noise breaks strict JSON / symbolic precision",
        },
        {
            "setting": "Non-causal aux roles",
            "finding": "Effective for summarization, tool selection, trajectory dedupe",
        },
    ]


def dllm_agent_findings() -> list[dict[str, str]]:
    return [
        {
            "metric": "End-to-end efficiency",
            "finding": "~30% average gain vs AR under matched interaction budget",
        },
        {
            "metric": "Task topology",
            "finding": "AR favors linear chains; dLLM favors constraint-intersection / reconciliation",
        },
        {
            "metric": "Failure mode",
            "finding": "Higher structured tool-call violation rate without structure-aware training",
        },
    ]


def p_react_paradigm() -> dict[str, Any]:
    return {
        "name": "Parallel-Reasoning and Acting (P-ReAct)",
        "paper": "2602.07035",
        "idea": "Decode tool_call instructions first; continue thinking during tool wait",
        "headline_latency_gain": "~15%",
        "training": ["Agentic SFT", "Agentic VRPO"],
    }


def deployment_ladder() -> list[dict[str, str]]:
    return [
        {"tier": "0", "mode": "AR-only", "when": "Strict JSON tools, short horizon"},
        {"tier": "1", "mode": "AR + dLLM summarize/select", "when": "Long chats, large tool catalogs"},
        {"tier": "2", "mode": "P-ReAct hybrid", "when": "Search agents with slow tool I/O"},
        {"tier": "3", "mode": "dLLM constraint planner", "when": "Multi-constraint reconciliation tasks"},
    ]


def taxonomy_bundle() -> dict[str, Any]:
    return {
        "bitter_lesson": bitter_lesson_findings(),
        "dllm_agent": dllm_agent_findings(),
        "p_react": p_react_paradigm(),
        "deployment_ladder": deployment_ladder(),
    }
