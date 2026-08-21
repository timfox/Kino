"""Paper table anchors and GPU benchmark excerpts (Sec. 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clairvoyant.constants import (
    PREDICTOR_LATENCY_MS_LMSYS,
    PREDICTOR_LATENCY_MS_SHAREGPT,
    UTIL_BENEFIT_RHO_MAX,
    UTIL_BENEFIT_RHO_MIN,
    UTIL_PEAK_RHO,
)
from ltx_trainer.clairvoyant.queueing import tau_sensitivity_table, utilisation_benefit_curve


def table5_ranking_vs_classification() -> list[dict[str, Any]]:
    """In-distribution ranking vs 3-class accuracy (Table 5)."""
    return [
        {"model": "A", "dataset": "ShareGPT", "ranking_acc_pct": 76.29, "class_acc_pct": 47.6},
        {"model": "B", "dataset": "LMSYS", "ranking_acc_pct": 95.62, "class_acc_pct": 66.8},
        {"model": "C", "dataset": "OASST1", "ranking_acc_pct": 62.21, "class_acc_pct": 41.0},
    ]


def table6_cross_distribution() -> dict[str, dict[str, float]]:
    """Cross-distribution ranking matrix (Table 6, off-diagonal true generalisation)."""
    return {
        "ShareGPT": {"ShareGPT": 86.4, "LMSYS": 53.6, "OASST1": 56.3, "Dolly": 52.7},
        "LMSYS": {"ShareGPT": 62.7, "LMSYS": 98.3, "OASST1": 65.3, "Dolly": 58.4},
        "OASST1": {"ShareGPT": 58.0, "LMSYS": 65.3, "OASST1": 90.4, "Dolly": 57.7},
    }


def table7_baselines() -> dict[str, dict[str, float]]:
    """Pairwise Short vs Long ranking for scheduling baselines (Table 7)."""
    return {
        "FCFS_random": {"ShareGPT": 50.0, "LMSYS": 50.0, "OASST1": 50.0},
        "prompt_length_rule": {"ShareGPT": 52.4, "LMSYS": 52.3, "OASST1": 55.8},
        "keyword_heuristic": {"ShareGPT": 36.3, "LMSYS": 4.6, "OASST1": 18.5},
        "clairvoyant_xgboost": {"ShareGPT": 74.9, "LMSYS": 95.1, "OASST1": 67.1},
    }


def table8_gpu_latency() -> list[dict[str, Any]]:
    """RTX 4090 burst benchmark (100 concurrent, n=250 per cell, Table 8)."""
    return [
        {
            "model": "Gemma3:4b",
            "class": "Short",
            "fcfs_p50": 229.5,
            "sjf_p50": 69.1,
            "fcfs_p95": 504.7,
            "sjf_p95": 163.0,
        },
        {
            "model": "Gemma3:4b",
            "class": "Long",
            "fcfs_p50": 309.7,
            "sjf_p50": 376.0,
            "fcfs_p95": 547.0,
            "sjf_p95": 554.9,
        },
        {
            "model": "Llama3.1:8b",
            "class": "Short",
            "fcfs_p50": 158.8,
            "sjf_p50": 38.0,
            "fcfs_p95": 352.0,
            "sjf_p95": 94.8,
        },
        {
            "model": "Llama3.1:8b",
            "class": "Long",
            "fcfs_p50": 188.7,
            "sjf_p50": 239.6,
            "fcfs_p95": 342.2,
            "sjf_p95": 355.3,
        },
    ]


def table4_ablation() -> list[dict[str, Any]]:
    """Feature ablation average ranking-accuracy delta (Table 4)."""
    return [
        {"feature_removed": "prompt_token_len", "avg_delta_pp": -3.09, "effect": "harmful"},
        {"feature_removed": "instruction_verb", "avg_delta_pp": -1.78, "effect": "mixed"},
        {"feature_removed": "has_code_keyword", "avg_delta_pp": -1.51, "effect": "harmful"},
        {"feature_removed": "has_format_keyword", "avg_delta_pp": +0.78, "effect": "net_harmful"},
        {"feature_removed": "clause_count", "avg_delta_pp": +1.07, "effect": "net_harmful"},
    ]


def service_time_stats_m1() -> list[dict[str, Any]]:
    """Table 1: Apple M1 Ollama Gemma3:4b service-time moments."""
    return [
        {"workload": "Short-only", "E_S_s": 2.1, "C2_s": 0.26, "holb_risk": "Low"},
        {"workload": "Long-only", "E_S_s": 29.7, "C2_s": 0.15, "holb_risk": "Low"},
        {"workload": "Mixed 50/50", "E_S_s": 15.9, "C2_s": 1.03, "holb_risk": "High"},
        {"workload": "Mixed 80/20", "E_S_s": 7.6, "C2_s": 2.59, "holb_risk": "Very high"},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "predictor_latency_ms": {
            "sharegpt_onnx": PREDICTOR_LATENCY_MS_SHAREGPT,
            "lmsys_onnx": PREDICTOR_LATENCY_MS_LMSYS,
        },
        "deployment_rho_band": {"min": UTIL_BENEFIT_RHO_MIN, "peak": UTIL_PEAK_RHO, "max": UTIL_BENEFIT_RHO_MAX},
        "table1_service_times_m1": service_time_stats_m1(),
        "table4_ablation": table4_ablation(),
        "table5_ranking": table5_ranking_vs_classification(),
        "table6_cross_distribution": table6_cross_distribution(),
        "table7_baselines": table7_baselines(),
        "table8_gpu_burst": table8_gpu_latency(),
        "table9_tau_sensitivity_simulated": tau_sensitivity_table(),
        "figure3_utilisation_curve": utilisation_benefit_curve(),
        "burst_short_p50_reduction_pct": {"Gemma3:4b": 70.0, "Llama3.1:8b": 76.0},
        "steady_state_short_p50_reduction_pct_at_rho_074": 17.0,
    }
