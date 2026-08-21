"""Paper tables and figure excerpts (arXiv:2605.18878)."""

from __future__ import annotations

from typing import Any


def table1_view_classifier_mlp() -> list[dict[str, Any]]:
    """Table 1 — MLP F1 (95% CI) per view, Day1–Day2 difference, no cross-lung pooling."""
    return [
        {"view": "Left-1", "f1": 0.53, "ci_low": 0.33, "ci_high": 0.73},
        {"view": "Left-2", "f1": 0.57, "ci_low": 0.37, "ci_high": 0.76},
        {"view": "Left-3", "f1": 0.68, "ci_low": 0.48, "ci_high": 0.84},
        {"view": "Right-1", "f1": 0.44, "ci_low": 0.24, "ci_high": 0.65},
        {"view": "Right-2", "f1": 0.47, "ci_low": 0.25, "ci_high": 0.68},
        {"view": "Right-3", "f1": 0.61, "ci_low": 0.40, "ci_high": 0.82},
        {"view": "All Views", "f1": 0.80, "ci_low": 0.62, "ci_high": 0.96},
    ]


def table1_all_views_by_classifier() -> list[dict[str, Any]]:
    """Table 1 — All Views row across classifiers."""
    return [
        {"classifier": "Decision-Tree", "f1": 0.53, "ci": "0.32–0.73"},
        {"classifier": "Random-Forest", "f1": 0.64, "ci": "0.40–0.85"},
        {"classifier": "SVM", "f1": 0.58, "ci": "0.36–0.80"},
        {"classifier": "MLP", "f1": 0.80, "ci": "0.62–0.96"},
        {"classifier": "MLP-Large", "f1": 0.75, "ci": "0.55–0.92"},
        {"classifier": "TabPFN", "f1": 0.55, "ci": "0.34–0.78"},
    ]


def fusion_heatmap_mlp() -> list[dict[str, Any]]:
    """Fig. 3 — fusion × temporal (MLP, All Views) F1 scores."""
    return [
        {"fusion": "Avg. Features", "concatenate": 0.45, "difference": 0.76},
        {"fusion": "Avg. Proba.", "concatenate": 0.55, "difference": 0.61},
        {"fusion": "Concatenate", "concatenate": 0.57, "difference": 0.80},
        {"fusion": "Max. Features", "concatenate": 0.49, "difference": 0.65},
        {"fusion": "Max. Votes", "concatenate": 0.55, "difference": 0.61},
    ]


def table2_day_pair_mlp() -> list[dict[str, Any]]:
    """Table 2 — MLP F1 under day-pair training settings (All Views)."""
    return [
        {
            "view": "All Views",
            "single_day_d1_d2": 0.80,
            "all_days_eval_d1_d2": 0.72,
            "all_days_eval_all_pairs": 0.72,
        },
        {
            "view": "Left-3",
            "single_day_d1_d2": 0.68,
            "all_days_eval_d1_d2": 0.44,
            "all_days_eval_all_pairs": 0.48,
        },
        {
            "view": "Right-3",
            "single_day_d1_d2": 0.61,
            "all_days_eval_d1_d2": 0.68,
            "all_days_eval_all_pairs": 0.65,
        },
    ]


def table3_biomarker_svm() -> list[dict[str, Any]]:
    """Table 3 excerpt — biomarker features (38-D) best SVM All Views Day1–Day2."""
    return [
        {"setting": "All Views (Day 1 vs. Day 2)", "classifier": "SVM", "f1": 0.71, "ci": "0.47–0.90"},
        {"setting": "All Views (Day 1 vs. Day 2)", "classifier": "MLP", "f1": 0.54, "ci": "0.34–0.73"},
        {"setting": "TSM embedding MLP (reference)", "classifier": "MLP", "f1": 0.80, "ci": "0.62–0.96"},
    ]


def ehr_biomarker_comparison() -> dict[str, Any]:
    """Fig. 2 left — 25-patient held-out with EHR records."""
    return {
        "cohort_n": 25,
        "clinical_ehr_f1": 0.75,
        "expert_lus_biomarker_f1": 0.75,
        "combined_f1": 0.877,
        "combined_f1_gain": 0.127,
        "combined_identifies_all_readmitted": True,
        "note": "Imaging biomarkers complement EHR features not captured in contemporary records.",
    }


def biomarker_selection_frequency() -> list[dict[str, Any]]:
    """Fig. 2 right — RF selection frequency (radial scale max ≈ 3.0)."""
    return [
        {"biomarker": "PL Location", "frequency": 3.0},
        {"biomarker": "B-Line", "frequency": 2.4},
        {"biomarker": "A-Line", "frequency": 2.2},
        {"biomarker": "PL Breaks", "frequency": 2.0},
        {"biomarker": "PL Indents", "frequency": 1.9},
        {"biomarker": "PL Thickness", "frequency": 1.7},
        {"biomarker": "B-Line Origin", "frequency": 1.5},
        {"biomarker": "Consolidation", "frequency": 1.2},
        {"biomarker": "Effusion", "frequency": 1.0},
    ]


def scan_counts_by_day() -> dict[str, int]:
    """CHF cohort scan counts by hospitalization day."""
    return {"day_1": 180, "day_2": 150, "day_3": 66, "day_4": 30}
