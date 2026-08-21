"""Paper table excerpts (arXiv:2605.28683)."""

from __future__ import annotations

from typing import Any


def table2_dataset_statistics() -> dict[str, Any]:
    """Table 2 — MRB, VKB, query difficulty distribution."""
    return {
        "mrb": {"documents": 8210, "images": 4146, "cities": 15},
        "vkb": {
            "transportation": 57773,
            "restaurants": 4304,
            "accommodations": 3613,
            "attractions": 1056,
            "pois": 7400,
        },
        "queries": {
            "simple": {"days": "2-5", "travelers": "1-2", "preferences": "1-4", "count": 78},
            "medium": {"days": "1-6", "travelers": "2-5", "preferences": "2-5", "count": 76},
            "complex": {"days": "3-7", "travelers": "2-8", "preferences": "2-8", "count": 74},
        },
        "topics": [
            "transportation",
            "restaurants",
            "accommodations",
            "attractions",
            "confusing_attractions",
            "travel_guides",
            "pois",
            "user_comments",
        ],
    }


def table3_evaluation_mapping() -> list[dict[str, str]]:
    """Table 3 — metric implementations."""
    return [
        {"dimension": "Format Check", "metric": "DR", "implementation": "parse_json(agent_output)"},
        {"dimension": "Fact: Transportation", "metric": "FR", "implementation": "exact_match(time, VKB)"},
        {"dimension": "Fact: POI", "metric": "FR", "implementation": "fuzzy_match(name, VKB)"},
        {"dimension": "Commonsense: Budget", "metric": "PRmi", "implementation": "total_cost <= budget * 1.5"},
        {"dimension": "Preference: Dining", "metric": "PFR", "implementation": "must_include(meals, preference)"},
        {"dimension": "Geographic", "metric": "AM", "implementation": "TSP_Solver vs agent route"},
    ]


def table4_main_results() -> list[dict[str, Any]]:
    """Table 4 excerpt — simple vs complex (headline models)."""
    return [
        {
            "model": "Claude-4.5-Sonnet",
            "simple": {"DR": 69.23, "FR": 62.81, "PFR": 64.10, "PRmi": 89.42, "AM": 8.49, "TC": 32},
            "complex": {"DR": 77.03, "FR": 68.60, "PFR": 50.00, "PRmi": 85.84, "AM": 16.59, "TC": 37},
        },
        {
            "model": "GPT-4o",
            "simple": {"DR": 63.64, "FR": 48.21, "PFR": 45.20, "PRmi": 88.02, "AM": 3.12, "TC": 8},
            "complex": {"DR": 69.57, "FR": 51.44, "PFR": 32.53, "PRmi": 80.12, "AM": 7.34, "TC": 11},
        },
        {
            "model": "o3",
            "simple": {"DR": 65.38, "FR": 46.81, "PFR": 55.38, "PRmi": 87.76, "AM": 8.93, "TC": 25},
            "complex": {"DR": 65.63, "FR": 53.43, "PFR": 27.57, "PRmi": 82.51, "AM": 10.04, "TC": 31},
        },
    ]


def table5_visual_ablation() -> list[dict[str, Any]]:
    """Table 5 — visual grounding ablation (GPT-4o excerpt)."""
    return [
        {"setting": "(qtxt,qimg) w/o Timg", "DR": 68.16, "FR": 48.79, "PFR": 43.86, "PRmi": 79.12},
        {"setting": "(qtxt,qimg) w Timg", "DR": 73.45, "FR": 56.12, "PFR": 42.50, "PRmi": 84.12},
        {"setting": "qtxt only (Gold POI)", "DR": 80.26, "FR": 52.60, "PFR": 63.16, "PRmi": 85.48},
    ]


def table6_noisy_mrb() -> list[dict[str, Any]]:
    """Table 6 — MRB vs MRB-Clean (GPT-4o)."""
    return [
        {"setting": "MRB-Clean", "DR": 74.32, "FR": 56.12, "PFR": 48.12, "PRmi": 87.55, "AM": 4.20},
        {"setting": "MRB", "DR": 73.45, "FR": 52.65, "PFR": 42.50, "PRmi": 84.12, "AM": 3.45},
    ]
