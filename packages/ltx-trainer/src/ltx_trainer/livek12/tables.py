"""Paper tables and headline metrics."""

from __future__ import annotations

from typing import Any


def figure1_degradation() -> list[dict[str, Any]]:
    """Default → +process+efficiency → full exam (×100 OES)."""
    return [
        {"model": "GPT-5", "default": 79.0, "process_efficiency": 64.0, "exam": 53.3},
        {"model": "Gemini-3-pro", "default": 88.7, "process_efficiency": 78.9, "exam": 58.4},
        {"model": "Claude-opus-4.6", "default": 88.7, "process_efficiency": 81.8, "exam": 61.8},
    ]


def table3_gemini_math_row() -> dict[str, float]:
    return {"Acc": 88.3, "ARL": 82.9, "PES": 88.3, "OES": 90.3}


def table3_gpt5_biology_row() -> dict[str, float]:
    return {"Acc": 71.6, "ARL": 70.3, "PES": 44.1, "OES": 53.7}


def table4_complex_layout_io_drop() -> list[dict[str, Any]]:
    return [
        {"model": "GPT-5", "acc_drop_pct": 32.3, "ocs_drop_pts": 44.0},
        {"model": "Gemini-3-pro", "acc_drop_pct": 28.3, "ocs_drop_pts": 38.9},
        {"model": "Kimi-k2.5", "acc_drop_pct": 26.0, "ocs_drop_pts": 38.4},
    ]


def table1_modality_counts() -> dict[str, int]:
    """Key statistics from paper Table 1 (approximate totals)."""
    return {
        "text_only": 1096,
        "text_image": 1018,
        "image_only_exam_mode": 2114,
        "total_questions": 2114,
    }


def headline_results() -> dict[str, str]:
    return {
        "data_leakage": "Dynamic ingestion of 2026 exam papers mitigates contamination vs static sets.",
        "visual": "Image-Only (IO) modality causes large Acc/OCS drops vs parsed TO/TI.",
        "process": "Rigorous-process subset: high OCS with low PES → lucky guesses.",
        "efficiency": "Long-horizon subset lowers ARL; token budgets mimic exam time limits.",
        "leader": "Gemini-3-pro leads OES; Claude-opus-4.6 leads ARL efficiency.",
    }
