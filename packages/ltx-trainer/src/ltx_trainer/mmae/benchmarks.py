"""Paper Table 1 statistics and Table 2 main results anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mmae.constants import (
    MMAE_AVG_CHOICES,
    MMAE_AVG_CR_RUBRICS,
    MMAE_AVG_DURATION_SEC,
    MMAE_AVG_IF_RUBRICS,
    MMAE_AVG_INSTRUCTION_WORDS,
    MMAE_AVG_OPERATIONS,
    MMAE_AVG_RUBRICS_PER_SAMPLE,
    MMAE_SHORT_SUBSET_COUNT,
    MMAE_TOTAL_RUBRICS,
    MMAE_TOTAL_SAMPLES,
)


def table1_statistics() -> dict[str, Any]:
    return {
        "total_samples": MMAE_TOTAL_SAMPLES,
        "total_rubrics": MMAE_TOTAL_RUBRICS,
        "avg_rubrics_per_sample": MMAE_AVG_RUBRICS_PER_SAMPLE,
        "avg_if_rubrics": MMAE_AVG_IF_RUBRICS,
        "avg_consistency_rubrics": MMAE_AVG_CR_RUBRICS,
        "avg_audio_duration_sec": MMAE_AVG_DURATION_SEC,
        "avg_instruction_words": MMAE_AVG_INSTRUCTION_WORDS,
        "avg_operations_per_sample": MMAE_AVG_OPERATIONS,
        "avg_choices_per_rubric": MMAE_AVG_CHOICES,
        "short_subset_leq_10s": MMAE_SHORT_SUBSET_COUNT,
    }


# Table 2(a) overall — percentages from paper
TABLE2_OVERALL: dict[str, dict[str, float]] = {
    "Identity": {"IFR": 27.37, "CR": 94.13, "EMR": 4.60},
    "Noise": {"IFR": 32.08, "CR": 15.68, "EMR": 0.00},
    "Step-Audio-EditX": {"IFR": 44.86, "CR": 58.88, "EMR": 3.05},
    "Ming-UniAudio": {"IFR": 29.82, "CR": 52.71, "EMR": 3.20},
    "MMEdit": {"IFR": 43.12, "CR": 47.64, "EMR": 3.50},
    "Audio-Omni": {"IFR": 50.73, "CR": 56.93, "EMR": 4.99},
    "SmartDJ w/o planner": {"IFR": 38.20, "CR": 55.41, "EMR": 4.62},
    "SmartDJ w/ planner": {"IFR": 42.26, "CR": 48.33, "EMR": 3.12},
}

TABLE2_BY_COMPLEXITY: dict[str, dict[str, dict[str, float]]] = {
    "Step-Audio-EditX": {
        "single": {"IFR": 46.64, "CR": 59.06, "EMR": 3.99},
        "multiple": {"IFR": 43.06, "CR": 58.69, "EMR": 2.11},
    },
    "Audio-Omni": {
        "single": {"IFR": 58.43, "CR": 64.57, "EMR": 6.25},
        "multiple": {"IFR": 41.70, "CR": 47.94, "EMR": 3.52},
    },
}

TABLE2_MIXED_MODALITY: dict[str, dict[str, float]] = {
    "Step-Audio-EditX": {"sound-music-speech": {"IFR": 44.73, "CR": 58.47, "EMR": 1.71}},
    "Ming-UniAudio": {"sound-music-speech": {"IFR": 26.93, "CR": 51.02, "EMR": 2.86}},
    "SmartDJ w/ planner": {"music-speech": {"IFR": 37.05, "CR": 46.09, "EMR": 0.00}},
}


def table2_main_results() -> dict[str, Any]:
    return {
        "overall": dict(TABLE2_OVERALL),
        "by_complexity": dict(TABLE2_BY_COMPLEXITY),
        "mixed_modality_highlights": dict(TABLE2_MIXED_MODALITY),
    }


def paper_claims() -> dict[str, bool]:
    editing = [m for m in TABLE2_OVERALL if m not in ("Identity", "Noise")]
    return {
        "all_emr_under_5pct": all(TABLE2_OVERALL[m]["EMR"] < 5.0 for m in editing),
        "audio_omni_best_ifr_short": TABLE2_OVERALL["Audio-Omni"]["IFR"] >= TABLE2_OVERALL["MMEdit"]["IFR"],
        "identity_high_cr": TABLE2_OVERALL["Identity"]["CR"] > 90.0,
        "noise_low_cr": TABLE2_OVERALL["Noise"]["CR"] < 20.0,
        "complexity_degrades_audio_omni": (
            TABLE2_BY_COMPLEXITY["Audio-Omni"]["single"]["IFR"]
            > TABLE2_BY_COMPLEXITY["Audio-Omni"]["multiple"]["IFR"]
        ),
        "planner_not_consistent_gain": TABLE2_OVERALL["SmartDJ w/ planner"]["CR"]
        < TABLE2_OVERALL["SmartDJ w/o planner"]["CR"],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1": table1_statistics(),
        "table2": table2_main_results(),
        "claims": paper_claims(),
    }
