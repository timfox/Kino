"""Paper result tables for TransFACT (arXiv:2605.18923)."""

from __future__ import annotations

from typing import Any


def table1_input_modalities() -> list[dict[str, Any]]:
    """Table 1 — TransFACT input configurations (mean ± std over 5 seeds)."""
    return [
        {
            "modality": "frames",
            "P_T": 80.11,
            "R_T": 84.94,
            "F1_T": 82.41,
            "P_NT": 85.52,
            "R_NT": 80.66,
            "F1_NT": 82.97,
            "accuracy": 82.70,
        },
        {
            "modality": "MHI",
            "P_T": 76.35,
            "R_T": 81.57,
            "F1_T": 78.86,
            "P_NT": 82.06,
            "R_NT": 76.92,
            "F1_NT": 79.40,
            "accuracy": 79.14,
        },
        {
            "modality": "frames+MHI",
            "P_T": 79.20,
            "R_T": 85.66,
            "F1_T": 82.21,
            "P_NT": 86.09,
            "R_NT": 79.34,
            "F1_NT": 82.46,
            "accuracy": 82.36,
        },
    ]


def table2_vs_sfr() -> list[dict[str, Any]]:
    """Table 2 — TransFACT (frames) vs SFR baseline."""
    return [
        {
            "model": "TransFACT",
            "F1_T": 82.41,
            "F1_NT": 82.97,
            "accuracy": 82.70,
        },
        {
            "model": "SFR",
            "F1_T": 69.08,
            "F1_NT": 68.75,
            "accuracy": 69.20,
        },
    ]


def table2_significance() -> dict[str, Any]:
    """Wilcoxon paired test summary (paper reports trend p<0.1, large Cohen's d)."""
    return {
        "test": "paired Wilcoxon signed-rank",
        "p_threshold_claimed": 0.05,
        "note": "Consistently favors TransFACT; several metrics p=0.063, Cohen d>0.8",
        "accuracy_cohens_d": 6.54,
    }


def progressive_accuracy_curve() -> list[dict[str, Any]]:
    """Fig. 3 key points — accuracy vs truncated video length."""
    return [
        {"frames": 15, "approx_dpi": 1, "accuracy_pct": 54},
        {"frames": 30, "event": "1st cleavage", "accuracy_pct": 65},
        {"frames": 60, "event": "2nd cleavage", "accuracy_pct": 70},
        {"frames": 90, "event": "3rd cleavage", "accuracy_pct": 74},
        {"frames": 120, "event": "LAG start", "accuracy_pct": 79},
        {"frames": 300, "event": "full 4 DPI", "accuracy_pct": 82},
    ]
