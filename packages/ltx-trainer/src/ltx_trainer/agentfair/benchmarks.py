"""Paper anchors and table stubs for AgentFAIR (Chen & Pai, arXiv:2607.15781)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "n_datasets": 50,
    "n_repositories": 10,
    "findability_mean": 79.7,
    "accessibility_mean": 70.4,
    "interoperability_mean": 45.3,
    "reusability_mean": 72.0,
    "cross_tool_stdev_mean": 15.0,
    "cross_tool_stdev_max": 30.3,
    "rho_fairshake": 0.61,
    "rho_fair_checker": 0.47,
    "rho_fuji": 0.42,
    "rho_fair_enough": 0.31,
    "repeat_agreement": 0.89,
    "repeat_stdev": 0.03,
    "critic_ablation_agreement": 0.71,
    "expert_kappa": 0.71,
    "expert_alignment": 0.82,
    "mean_cost_usd": 0.054,
    "total_cost_usd": 2.68,
    "mean_seconds": 1054,
    "critic_trigger_rate": 0.96,
    "n_subprinciples": 13,
}

# Table 4 case-study dimension scores (%) from the paper
TABLE_4_CASE_STUDIES: list[dict[str, Any]] = [
    {
        "dataset": "AURIN OSM POIs (D2)",
        "F": 41.7,
        "A": 66.7,
        "I": 0.0,
        "R": 66.7,
        "overall": 43.6,
    },
    {
        "dataset": "NASA GDIS (D3)",
        "F": 100.0,
        "A": 77.8,
        "I": 44.4,
        "R": 55.6,
        "overall": 71.8,
    },
    {
        "dataset": "Zenodo Crater Lake (D1)",
        "F": 83.3,
        "A": 77.8,
        "I": 55.6,
        "R": 77.8,
        "overall": 74.4,
    },
]

TABLE_6_SPEARMAN: list[dict[str, Any]] = [
    {"baseline": "F-UJI", "n": 50, "rho": 0.42, "p": 0.002},
    {"baseline": "FAIR-Checker", "n": 49, "rho": 0.47, "p": 0.001},
    {"baseline": "FAIRshake", "n": 50, "rho": 0.61, "p": 0.001},
    {"baseline": "FAIR-enough", "n": 32, "rho": 0.31, "p": 0.084},
]

TABLE_7_EFFICIENCY: dict[str, Any] = {
    "total_tokens_m": 16.54,
    "total_cost_usd": 2.68,
    "mean_cost_usd": 0.054,
    "mean_processing_seconds": 1054,
}

TABLE_8_CRITIC_ABLATION: list[dict[str, Any]] = [
    {
        "configuration": "Full system (with critic)",
        "consistency": "89±3%",
        "reeval_rate": "96% triggered",
    },
    {
        "configuration": "Without critic agent",
        "consistency": "71±5%",
        "reeval_rate": "N/A",
    },
]

REPOSITORY_COUNTS: dict[str, int] = {
    "Zenodo": 10,
    "Dryad": 8,
    "PANGAEA": 7,
    "Harvard Dataverse": 6,
    "NOAA NCEI": 6,
    "ScienceBase": 4,
    "Figshare": 4,
    "AURIN": 2,
    "EarthData": 2,
    "NASA SEDAC": 1,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_4_case_studies": list(TABLE_4_CASE_STUDIES),
        "table_6_spearman": list(TABLE_6_SPEARMAN),
        "table_7_efficiency": dict(TABLE_7_EFFICIENCY),
        "table_8_critic_ablation": list(TABLE_8_CRITIC_ABLATION),
        "repository_counts": dict(REPOSITORY_COUNTS),
    }
