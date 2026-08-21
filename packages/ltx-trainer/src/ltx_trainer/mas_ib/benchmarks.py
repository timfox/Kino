"""Paper anchors and table stubs for MAS-IB (arXiv:2607.16133)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "system": "MAS-IB",
    "n_controlled_experiments": 18,
    "n_benchmarks": 5,
    "n_model_scales": 3,
    "alfworld_qwen7b_gain": 0.194,
    "alfworld_qwen27b_gain": 0.023,
    "workbench_gpt4o_gain": -0.086,
    "travelplanner_hc_qwen27b_gain": -0.233,
    "webshop_qwen27b_gain": -0.003,
    "github": "https://github.com/divelab/MAS-SAS",
}

# Table 2: MAS − SAS-contextflow by δ regime
TABLE_2_GAINS: list[dict[str, Any]] = [
    {
        "task": "ALFWorld",
        "delta_regime": "≈0",
        "qwen25_7b": 0.194,
        "gpt4o_mini": 0.157,
        "qwen35_27b": 0.023,
    },
    {
        "task": "WideSearch",
        "delta_regime": "≈0",
        "qwen25_7b": 0.079,
        "gpt4o_mini": 0.063,
        "qwen35_27b": 0.028,
    },
    {
        "task": "TravelPlanner-CS",
        "delta_regime": "≈0",
        "qwen25_7b": 0.011,
        "gpt4o_mini": 0.183,
        "qwen35_27b": 0.028,
    },
    {
        "task": "WebShop",
        "delta_regime": ">0",
        "qwen25_7b": 0.080,
        "gpt4o_mini": 0.086,
        "qwen35_27b": -0.003,
    },
    {
        "task": "WorkBench",
        "delta_regime": "≫0",
        "qwen25_7b": -0.005,
        "gpt4o_mini": -0.086,
        "qwen35_27b": -0.014,
    },
    {
        "task": "TravelPlanner-HC",
        "delta_regime": ">0",
        "qwen25_7b": 0.017,
        "gpt4o_mini": 0.161,
        "qwen35_27b": -0.233,
    },
]

TABLE_1_DELTA_REGIMES: list[dict[str, Any]] = [
    {"dataset": "ALFWorld", "delta": "≈0", "property": "Needs only current object state and location"},
    {"dataset": "WideSearch", "delta": "≈0", "property": "Subqueries mostly independent"},
    {"dataset": "TravelPlanner-CS", "delta": "≈0", "property": "Commonsense checks mostly local"},
    {"dataset": "WebShop", "delta": ">0", "property": "Purchase needs accumulated product evidence"},
    {"dataset": "WorkBench", "delta": "≫0", "property": "Downstream needs exact prior states/outputs"},
    {"dataset": "TravelPlanner-HC", "delta": ">0", "property": "Hard constraints need global consistency"},
]

TABLE_6_ALFWORLD_ZEROSHOT: list[dict[str, Any]] = [
    {"model": "Qwen2.5-7B", "SAS": 0.366, "SAS_contextflow": 0.172, "MAS": 0.545},
    {"model": "GPT-4o-mini", "SAS": 0.507, "SAS_contextflow": 0.373, "MAS": 0.552},
    {"model": "Qwen3.5-27B", "SAS": 0.903, "SAS_contextflow": 0.888, "MAS": 0.895},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_1_delta_regimes": list(TABLE_1_DELTA_REGIMES),
        "table_2_gains": list(TABLE_2_GAINS),
        "table_6_alfworld_zeroshot": list(TABLE_6_ALFWORLD_ZEROSHOT),
    }
