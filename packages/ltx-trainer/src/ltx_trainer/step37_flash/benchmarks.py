"""Benchmark anchors — Step 3.7 Flash (StepFun 2026)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.step37_flash.constants import (
    BENCHMARK_SCORES,
    GITHUB_REPO,
    MODEL_PAGE,
    PRICING_PER_M,
)

# Multimodal (README)
TABLE_MULTIMODAL: dict[str, float] = {
    "SimpleVQA_Search": BENCHMARK_SCORES["SimpleVQA_Search"],
    "V_star_Python": BENCHMARK_SCORES["V_star_Python"],
}

# Agent / tools
TABLE_AGENT: dict[str, float] = {
    "ClawEval_1_1": BENCHMARK_SCORES["ClawEval_1_1"],
    "ClawEval_runner_up": BENCHMARK_SCORES["ClawEval_runner_up"],
    "Toolathlon": BENCHMARK_SCORES["Toolathlon"],
    "HLE_w_Tool": BENCHMARK_SCORES["HLE_w_Tool"],
}

# Coding / professional
TABLE_CODING: dict[str, float] = {
    "SWE_Bench_PRO": BENCHMARK_SCORES["SWE_Bench_PRO"],
    "Terminal_Bench_2_1": BENCHMARK_SCORES["Terminal_Bench_2_1"],
    "GDPVal_AA": BENCHMARK_SCORES["GDPVal_AA"],
}

# Competitive context (blog claims)
COMPARATIVE_NOTES = {
    "SimpleVQA_Search": "1st place",
    "ClawEval_1_1": "leads next competitor by 7.3 points",
    "SWE_Bench_PRO": "definitive 2nd place",
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "model_page": MODEL_PAGE,
        "github": GITHUB_REPO,
        "multimodal": TABLE_MULTIMODAL,
        "agent_tools": TABLE_AGENT,
        "coding_professional": TABLE_CODING,
        "comparative_notes": COMPARATIVE_NOTES,
        "pricing_per_m_usd": PRICING_PER_M,
    }
