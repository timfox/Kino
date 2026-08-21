"""Paper table anchors — InstructAV2AV (arXiv:2605.18467)."""

from __future__ import annotations

from typing import Any

# Table 1 — InsAVE-80K (paper values)
TABLE1_INSAVE: dict[str, dict[str, float]] = {
    "InstructAV2AV": {
        "FVD": 180.38,
        "TV-A": 25.23,
        "TC": 95.65,
        "SSIM": 93.84,
        "FAD": 2.75,
        "TA-A": 34.96,
        "LPAPS": 1.77,
        "AV-A": 27.72,
        "PEAVS": 3.24,
        "S-C": 4.67,
        "S-D": 8.96,
    },
    "AVI-Edit": {
        "FVD": 205.34,
        "TV-A": 25.15,
        "TC": 95.45,
        "SSIM": 90.66,
        "FAD": 5.09,
        "TA-A": 30.07,
        "LPAPS": 1.87,
        "AV-A": 26.37,
        "PEAVS": 2.54,
        "S-C": 4.06,
        "S-D": 9.25,
    },
    "AvED": {
        "FVD": 305.41,
        "TV-A": 24.68,
        "TC": 94.73,
        "SSIM": 90.78,
        "FAD": 2.86,
        "TA-A": 29.89,
        "LPAPS": 1.80,
        "AV-A": 26.44,
        "PEAVS": 3.15,
        "S-C": 2.98,
        "S-D": 9.42,
    },
}

TABLE1_AVED_BENCH: dict[str, dict[str, float]] = {
    "InstructAV2AV": {
        "FVD": 227.82,
        "TV-A": 24.26,
        "TC": 95.27,
        "SSIM": 92.72,
        "FAD": 4.32,
        "TA-A": 35.89,
        "LPAPS": 1.58,
        "AV-A": 23.71,
        "PEAVS": 3.20,
    },
}

# Table 1 ablations (InsAVE-80K)
TABLE1_ABLATION: dict[str, dict[str, float]] = {
    "w/o_SC": {"FVD": 467.20, "SSIM": 49.97, "TA-A": 34.00},
    "w/o_SIGA": {"FVD": 187.28, "SSIM": 93.65, "TA-A": 34.50},
    "w/o_TSTS": {"FVD": 291.55, "SSIM": 85.63, "TA-A": 34.80},
    "InstructAV2AV": {"FVD": 180.38, "SSIM": 93.84, "TA-A": 34.96},
}

# Table 2 — user preference % (InsAVE-80K)
TABLE2_USER_PREF: dict[str, dict[str, float]] = {
    "InstructAV2AV": {"AVS": 49.0, "TA": 45.4, "OP": 46.6},
    "AVI-Edit": {"AVS": 32.6, "TA": 31.4, "OP": 34.0},
    "CoherentAVEdit": {"AVS": 17.0, "TA": 20.6, "OP": 18.6},
    "AvED": {"AVS": 1.4, "TA": 2.6, "OP": 0.8},
}

# Table 3 — capability flags (Appendix 7.1)
TABLE3_CAPABILITIES: dict[str, dict[str, bool]] = {
    "InstructAV2AV": {
        "train_based": True,
        "end_to_end": True,
        "mask_free": True,
        "text_guided": True,
        "human_speech": True,
        "general_audio": True,
    },
    "AVI-Edit": {
        "train_based": True,
        "end_to_end": True,
        "mask_free": False,
        "text_guided": True,
        "human_speech": True,
        "general_audio": True,
    },
    "AvED": {
        "train_based": False,
        "end_to_end": False,
        "mask_free": True,
        "text_guided": True,
        "human_speech": False,
        "general_audio": True,
    },
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_insave": TABLE1_INSAVE,
        "table1_aved_bench": TABLE1_AVED_BENCH,
        "table1_ablation": TABLE1_ABLATION,
        "table2_user_preference": TABLE2_USER_PREF,
        "table3_capabilities": TABLE3_CAPABILITIES,
        "metrics_count": 11,
        "eval_sets": ["InsAVE-80K", "AvED-Bench"],
        "website": "https://hjzheng.net/projects/InstructAV2AV/",
    }
