"""Paper tables and generator failure-rate excerpts (MTAVG-Bench 2.0)."""

from __future__ import annotations

from typing import Any

# Table 3 — omni model diagnosis accuracy by sub-dimension (%), paper-printed values.
TABLE3_OMNI_DIAGNOSIS_PCT: dict[str, dict[str, float]] = {
    "Gemini 3.1 Flash Lite": {
        "EP": 48.66,
        "MP": 27.62,
        "DP": 49.64,
        "IP": 38.08,
        "MC": 32.87,
        "EC": 40.84,
        "SD": 66.67,
        "IC": 33.30,
        "IG": 49.42,
        "CT": 56.97,
        "Avg": 44.41,
    },
    "Gemini 3.1 Pro": {
        "EP": 53.49,
        "MP": 52.35,
        "DP": 81.77,
        "IP": 43.00,
        "MC": 71.86,
        "EC": 51.57,
        "SD": 70.62,
        "IC": 49.26,
        "IG": 72.68,
        "CT": 75.05,
        "Avg": 62.16,
    },
    "Gemini 3 Flash": {
        "EP": 46.11,
        "MP": 32.21,
        "DP": 76.45,
        "IP": 43.57,
        "MC": 65.21,
        "EC": 46.24,
        "SD": 59.72,
        "IC": 44.85,
        "IG": 59.21,
        "CT": 66.40,
        "Avg": 54.00,
    },
    "Gemini 2.5 Flash": {
        "EP": 31.68,
        "MP": 34.05,
        "DP": 43.22,
        "IP": 34.84,
        "MC": 64.36,
        "EC": 45.32,
        "SD": 65.17,
        "IC": 36.91,
        "IG": 56.36,
        "CT": 58.19,
        "Avg": 47.01,
    },
    "Qwen 2.5 Omni 7B": {
        "EP": 35.52,
        "MP": 23.86,
        "DP": 40.80,
        "IP": 33.44,
        "MC": 34.75,
        "EC": 31.28,
        "SD": 45.62,
        "IC": 32.90,
        "IG": 51.58,
        "CT": 48.20,
        "Avg": 37.80,
    },
    "Ming Lite Omni 1.5 30B": {
        "EP": 49.35,
        "MP": 37.24,
        "DP": 44.96,
        "IP": 33.38,
        "MC": 32.40,
        "EC": 36.38,
        "SD": 41.13,
        "IC": 31.23,
        "IG": 55.69,
        "CT": 49.30,
        "Avg": 41.11,
    },
}

# Table 4 — holistic automatic metrics (paper values).
TABLE4_HOLISTIC_METRICS: dict[str, dict[str, float]] = {
    "Grok Video 3": {
        "AudioAesthetic": 4.283,
        "LipSync": 0.682,
        "AVAlign": 0.160,
        "Desync": 0.614,
        "TAAlign": 0.349,
        "TVAlign": 0.222,
    },
    "LTX 2.3": {
        "AudioAesthetic": 3.942,
        "LipSync": 0.454,
        "AVAlign": 0.117,
        "Desync": 0.377,
        "TAAlign": 0.170,
        "TVAlign": 0.191,
    },
    "Sora 2": {
        "AudioAesthetic": 2.614,
        "LipSync": 0.438,
        "AVAlign": 0.187,
        "Desync": 0.593,
        "TAAlign": 0.158,
        "TVAlign": 0.215,
    },
    "Veo 3.1": {
        "AudioAesthetic": 3.626,
        "LipSync": 1.116,
        "AVAlign": 0.235,
        "Desync": 0.600,
        "TAAlign": 0.237,
        "TVAlign": 0.211,
    },
    "Vidu Q3": {
        "AudioAesthetic": 4.108,
        "LipSync": 0.952,
        "AVAlign": 0.194,
        "Desync": 0.635,
        "TAAlign": 0.204,
        "TVAlign": 0.204,
    },
    "Wan 2.6": {
        "AudioAesthetic": 4.241,
        "LipSync": 0.458,
        "AVAlign": 0.128,
        "Desync": 0.545,
        "TAAlign": 0.260,
        "TVAlign": 0.211,
    },
}

# Fig. 4 — reference failure rates (%) per sub-dimension for generators (paper narrative + relative ordering).
# LTX 2.3 called out as persistently weak on scene-level coordination dimensions.
FIG4_FAILURE_RATE_PCT: dict[str, dict[str, float]] = {
    "LTX 2.3": {
        "EP": 0.38,
        "MP": 0.42,
        "DP": 0.55,
        "IP": 0.52,
        "MC": 0.48,
        "EC": 0.35,
        "SD": 0.32,
        "IC": 0.40,
        "IG": 0.50,
        "CT": 0.47,
        "average": 0.44,
    },
    "Grok Video 3": {
        "EP": 0.36,
        "MP": 0.40,
        "DP": 0.50,
        "IP": 0.48,
        "MC": 0.46,
        "EC": 0.42,
        "SD": 0.30,
        "IC": 0.38,
        "IG": 0.48,
        "CT": 0.45,
        "average": 0.43,
    },
    "Wan 2.6": {
        "EP": 0.30,
        "MP": 0.34,
        "DP": 0.42,
        "IP": 0.40,
        "MC": 0.38,
        "EC": 0.32,
        "SD": 0.28,
        "IC": 0.35,
        "IG": 0.40,
        "CT": 0.38,
        "average": 0.36,
    },
}

# Table 1 comparison row (benchmark paradigm).
TABLE1_BENCHMARK_PARADIGM: dict[str, dict[str, Any]] = {
    "MTAVG-Bench": {
        "videos": 1880,
        "qa": 2410,
        "dimensions": 9,
        "failure_modes": 37,
        "cinematic_expressiveness": False,
    },
    "MTAVG-Bench 2.0": {
        "videos": 2466,
        "qa": 11600,
        "dimensions": 10,
        "failure_modes": 45,
        "cinematic_expressiveness": True,
    },
}

# Table 5 — temporal localization (Gemini 3.1 Pro).
TABLE5_TEMPORAL_LOCALIZATION: dict[str, float] = {
    "PIA_pct": 60.6,
    "TLA_pct": 60.9,
    "RC_pct": 83.8,
}

# Table 6 — Gemini 3 Flash modality ablation (%).
TABLE6_MODALITY_ABLATION: dict[str, dict[str, float]] = {
    "Full Input": {"acting": 49.58, "atmosphere": 57.06, "cinematography": 56.82, "avg": 54.00},
    "w/o Audio": {"acting": 36.40, "atmosphere": 41.20, "cinematography": 45.73, "avg": 40.64},
    "w/o Vision": {"acting": 36.67, "atmosphere": 33.70, "cinematography": 27.73, "avg": 33.10},
    "Text-only": {"acting": 34.88, "atmosphere": 33.10, "cinematography": 29.80, "avg": 32.82},
}

# Fig. 3 question counts per sub-dimension (approximate from paper figure).
FIG3_QUESTION_COUNTS: dict[str, int] = {
    "EP": 411,
    "MP": 1191,
    "DP": 1118,
    "IP": 1573,
    "MC": 1813,
    "EC": 1315,
    "SD": 156,
    "IC": 1643,
    "IG": 1111,
    "CT": 1141,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_paradigm": TABLE1_BENCHMARK_PARADIGM,
        "table3_omni_diagnosis_pct": TABLE3_OMNI_DIAGNOSIS_PCT,
        "table4_holistic_metrics": TABLE4_HOLISTIC_METRICS,
        "table5_temporal_localization": TABLE5_TEMPORAL_LOCALIZATION,
        "table6_modality_ablation": TABLE6_MODALITY_ABLATION,
        "fig3_question_counts": FIG3_QUESTION_COUNTS,
        "fig4_failure_rate_pct": FIG4_FAILURE_RATE_PCT,
        "ltx23_holistic": TABLE4_HOLISTIC_METRICS["LTX 2.3"],
        "gemini31_pro_avg_diagnosis": TABLE3_OMNI_DIAGNOSIS_PCT["Gemini 3.1 Pro"]["Avg"],
    }
