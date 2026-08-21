"""Paper anchors and table stubs for DiffTestGen (arXiv:2607.16024)."""

from __future__ import annotations

from typing import Any

PAPER_ANCHORS: dict[str, float | int | str] = {
    "system": "DiffTestGen",
    "n_prs": 463,
    "overall_prs_pct": 0.782,
    "overall_union_coverage": 0.907,
    "testora_data_prs": 439,
    "testora_difftestgen_prs": 350,
    "testora_union_coverage": 0.927,
    "chaco_data_prs": 34,
    "chaco_gpt5_prs": 28,
    "chaco_gpt5_union": 0.768,
    "chaco_gpt4o_union": 0.645,
    "baseline_extra_prs": 99,
    "coverage_gain_pp_a": 0.125,
    "coverage_gain_pp_b": 0.156,
    "regressions_confirmed": 5,
    "github": "https://github.com/sola-st/DiffTestGen",
}

# Fig. 5 / overall Testora-data comparison
TABLE_TESTORA_OVERALL: dict[str, dict[str, float | int]] = {
    "Testora": {"num_pr": 251, "union_coverage": 0.771},
    "Testora++": {"num_pr": 277, "union_coverage": 0.807},
    "Only Coverage Feedback": {"num_pr": 261, "union_coverage": 0.822},
    "Only Access Information": {"num_pr": 330, "union_coverage": 0.858},
    "DiffTestGen": {"num_pr": 350, "union_coverage": 0.927},
}

TABLE_CHACO: list[dict[str, Any]] = [
    {
        "project": "pandas",
        "chaco_union": 0.487,
        "chaco_prs": 10,
        "dtg_gpt4o_union": 0.643,
        "dtg_gpt4o_prs": 11,
        "dtg_gpt5_union": 0.834,
        "dtg_gpt5_prs": 15,
    },
    {
        "project": "scipy",
        "chaco_union": 0.549,
        "chaco_prs": 11,
        "dtg_gpt4o_union": 0.648,
        "dtg_gpt4o_prs": 10,
        "dtg_gpt5_union": 0.709,
        "dtg_gpt5_prs": 13,
    },
    {
        "project": "Total",
        "chaco_union": 0.520,
        "chaco_prs": 21,
        "dtg_gpt4o_union": 0.645,
        "dtg_gpt4o_prs": 21,
        "dtg_gpt5_union": 0.768,
        "dtg_gpt5_prs": 28,
    },
]

TABLE_COST: list[dict[str, Any]] = [
    {"approach": "Testora", "tokens": 10617, "dollars": 0.018, "minutes": 8.92},
    {"approach": "Testora++", "tokens": 24115, "dollars": 0.045, "minutes": 25.36},
    {"approach": "DiffTestGen", "tokens": 31260, "dollars": 0.041, "minutes": 15.86},
]

TABLE_GENERATED_TESTS: list[dict[str, Any]] = [
    {"approach": "Testora", "total": 14129, "num_diff_tests": 2193, "pct": 0.155},
    {"approach": "Testora++", "total": 66948, "num_diff_tests": 9107, "pct": 0.136},
    {"approach": "DiffTestGen", "total": 15055, "num_diff_tests": 4189, "pct": 0.278},
]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "anchors": dict(PAPER_ANCHORS),
        "table_testora_overall": dict(TABLE_TESTORA_OVERALL),
        "table_chaco": list(TABLE_CHACO),
        "table_cost": list(TABLE_COST),
        "table_generated_tests": list(TABLE_GENERATED_TESTS),
    }
