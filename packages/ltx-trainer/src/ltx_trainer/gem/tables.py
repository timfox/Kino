"""Paper table excerpts (arXiv:2605.26121)."""

from __future__ import annotations

from typing import Any


def table1_main_average() -> list[dict[str, Any]]:
    """Table 1 — average downstream accuracy (%) under mixing frameworks."""
    frameworks = ("DoReMi", "Perf", "RegMix")
    rows = []
    for fw in frameworks:
        rows.append(
            {
                "framework": fw,
                "kmeans_avg": {"DoReMi": 39.94, "Perf": 41.04, "RegMix": 40.72}[fw],
                "tos_avg": {"DoReMi": 40.25, "Perf": 40.11, "RegMix": 39.64}[fw],
                "weborg_topic_avg": {"DoReMi": 42.76, "Perf": 44.23, "RegMix": 40.08}[fw],
                "gem_avg": {"DoReMi": 43.95, "Perf": 44.79, "RegMix": 41.45}[fw],
            }
        )
    return rows


def table1_gem_breakdown() -> dict[str, float]:
    """GEM (Ours) row highlights under DoReMi / Perf."""
    return {
        "science_qa_doremi": 34.79,
        "commonsense_doremi": 39.96,
        "logic_doremi": 57.11,
        "average_doremi": 43.95,
        "science_qa_perf": 35.96,
        "commonsense_perf": 40.43,
        "logic_perf": 57.98,
        "average_perf": 44.79,
    }


def ablation_clustering_avg() -> list[dict[str, Any]]:
    """Figure 6 — ablation on clustering mechanisms (Perf setting)."""
    return [
        {"method": "K-Means", "science_qa": 32.5, "commonsense": 35.1, "logic": 55.5, "average": 38.5},
        {"method": "Spherical K-Means", "science_qa": 34.6, "commonsense": 40.0, "logic": 53.9, "average": 40.6},
        {"method": "Vanilla vMF", "science_qa": 34.5, "commonsense": 40.2, "logic": 56.3, "average": 41.3},
        {"method": "GEM (Ours)", "science_qa": 36.0, "commonsense": 40.4, "logic": 57.0, "average": 42.1},
    ]


def k_sensitivity_perf() -> list[dict[str, Any]]:
    """Figure 5 — accuracy vs K (GEM + Perf)."""
    ks = [12, 24, 36, 48]
    science = [33.9, 35.8, 37.5, 38.6]
    commonsense = [33.6, 35.2, 39.5, 38.1]
    logic = [53.5, 56.7, 54.4, 52.9]
    avg = [39.5, 39.9, 41.2, 40.6]
    return [
        {
            "k": k,
            "science_qa": science[i],
            "commonsense": commonsense[i],
            "logic": logic[i],
            "average": avg[i],
        }
        for i, k in enumerate(ks)
    ]


def student_distillation_accuracy() -> dict[str, float]:
    """Appendix E — FastText student on cluster labels."""
    return {
        "kmeans_student_accuracy": 0.7292,
        "gem_student_accuracy": 0.7513,
    }


def headline_results() -> dict[str, Any]:
    return {
        "max_average_gain_pts": 1.2,
        "best_k_perf": 36,
        "best_average_at_k36": 41.21,
        "gem_vs_weborg_doremi_avg_delta": 1.19,
    }
