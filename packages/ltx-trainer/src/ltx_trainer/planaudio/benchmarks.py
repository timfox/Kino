"""Paper tables (PlanAudio-Bench, AudioCaps, LibriTTS excerpts)."""

from __future__ import annotations

from typing import Any

# Table 2 — Composite on PlanAudio-Bench
TABLE2_COMPOSITE: dict[str, dict[str, float]] = {
    "GroundTruth": {"FAD_PANNs": 0.00, "FAD_PaSST": 0.00, "KL_PaSST": 0.00, "KL_PANNs": 0.00, "IS": 3.23, "CLAP": 0.17, "WER": 0.10, "UTMOS": 2.69},
    "VoiceLDM-s": {"FAD_PANNs": 25.2, "FAD_PaSST": 379, "KL_PaSST": 1.39, "KL_PANNs": 1.53, "IS": 2.71, "CLAP": 0.15, "WER": 0.70, "UTMOS": 2.35},
    "VoiceLDM-m": {"FAD_PANNs": 22.9, "FAD_PaSST": 363, "KL_PaSST": 1.32, "KL_PANNs": 1.41, "IS": 2.86, "CLAP": 0.19, "WER": 0.09, "UTMOS": 2.81},
    "Pipeline_AudioLDM2": {"FAD_PANNs": 14.3, "FAD_PaSST": 240, "KL_PaSST": 1.10, "KL_PANNs": 1.15, "IS": 4.11, "CLAP": 0.21, "WER": 0.71, "UTMOS": 2.16},
    "PlanAudio": {"FAD_PANNs": 8.52, "FAD_PaSST": 201, "KL_PaSST": 0.91, "KL_PANNs": 1.03, "IS": 3.43, "CLAP": 0.20, "WER": 0.41, "UTMOS": 2.43},
}

# Table 3 — Subjective composite (0–5)
TABLE3_SUBJECTIVE: dict[str, dict[str, float]] = {
    "VoiceLDM-s": {"Quality": 2.78, "Temporal": 2.65, "Semantic": 2.67, "Authenticity": 2.71},
    "VoiceLDM-m": {"Quality": 2.83, "Temporal": 2.78, "Semantic": 2.95, "Authenticity": 2.93},
    "Pipeline_AudioLDM2": {"Quality": 2.24, "Temporal": 2.20, "Semantic": 2.38, "Authenticity": 2.40},
    "PlanAudio": {"Quality": 3.23, "Temporal": 3.16, "Semantic": 3.36, "Authenticity": 3.47},
}

# Table 4 — Sound (AudioCaps) — selected rows
TABLE4_SOUND: dict[str, dict[str, float]] = {
    "AudioLDM2": {"FAD_PANNs": 32.5, "KL_PaSST": 1.56, "IS": 8.54, "CLAP": 0.21},
    "VoiceLDM-m": {"FAD_PANNs": 55.8, "KL_PaSST": 3.37, "IS": 4.18, "CLAP": 0.07},
    "PlanAudio": {"FAD_PANNs": 24.7, "KL_PaSST": 1.93, "IS": 8.02, "CLAP": 0.19},
}

# Table 5 — Speech (LibriTTS)
TABLE5_SPEECH: dict[str, dict[str, float]] = {
    "PromptTTS++": {"WER": 0.12, "UTMOS": 3.51},
    "VoiceLDM-m": {"WER": 0.13, "UTMOS": 2.99},
    "PlanAudio": {"WER": 0.11, "UTMOS": 3.11},
}

# Table 6 — CoT ablation (subset training)
TABLE6_COT: dict[str, dict[str, float]] = {
    "PlanAudio": {"FD": 177, "KL": 1.07, "CLAP": 0.20, "WER": 0.86, "UTMOS": 2.24, "SCF": 0.34},
    "w/o CoT": {"FD": 217, "KL": 1.43, "CLAP": 0.15, "WER": 0.92, "UTMOS": 2.16, "SCF": 0.12},
    "Explicit CoT": {"FD": 230, "KL": 1.38, "CLAP": 0.16, "WER": 1.12, "UTMOS": 2.13, "SCF": 0.20},
    "Acoustic CoT": {"FD": 319, "KL": 1.37, "CLAP": 0.14, "WER": 1.21, "UTMOS": 2.24, "SCF": 0.09},
}

# Table 7 — curriculum sampling weights (sound/speech/composite)
TABLE7_CURRICULUM: dict[str, dict[str, tuple[float, float, float]]] = {
    "Constant": {
        "early": (0.33, 0.33, 0.33),
        "middle": (0.33, 0.33, 0.33),
        "final": (0.33, 0.33, 0.33),
    },
    "Gradual": {
        "early": (0.40, 0.40, 0.20),
        "middle": (0.40, 0.20, 0.40),
        "final": (0.25, 0.25, 0.50),
    },
    "Disjoint": {
        "early": (0.50, 0.50, 0.00),
        "middle": (0.50, 0.50, 0.00),
        "final": (0.00, 0.00, 1.00),
    },
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table2_composite": TABLE2_COMPOSITE,
        "table3_subjective": TABLE3_SUBJECTIVE,
        "table4_sound": TABLE4_SOUND,
        "table5_speech": TABLE5_SPEECH,
        "table6_cot": TABLE6_COT,
        "table7_curriculum": TABLE7_CURRICULUM,
        "planaudio_wins_composite_fad_panns": TABLE2_COMPOSITE["PlanAudio"]["FAD_PANNs"]
        < TABLE2_COMPOSITE["VoiceLDM-m"]["FAD_PANNs"],
        "planaudio_best_subjective_authenticity": TABLE3_SUBJECTIVE["PlanAudio"]["Authenticity"],
    }
