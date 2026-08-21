"""Paper table anchors and ablation bundles."""

from __future__ import annotations

TABLE1_MARCEL = {
    "PrismAvatar": {
        "O_alpha": 0.0691,
        "G_m": 0.0237,
        "G_d": 0.0297,
        "S_n": 0.0304,
        "S_alpha": 0.0149,
        "F_c": 0.2145,
    },
    "RGBAvatar": {
        "O_alpha": 0.0768,
        "G_m": 0.0303,
        "G_d": 0.0384,
        "S_n": 0.0357,
        "S_alpha": 0.0182,
        "F_c": 0.2117,
    },
    "HRAvatar": {
        "O_alpha": 0.0889,
        "G_m": 0.0314,
        "G_d": 0.0397,
        "S_n": 0.0177,
        "S_alpha": 0.0085,
        "F_c": 0.2017,
    },
    "FlashAvatar": {
        "O_alpha": 0.0746,
        "G_m": 0.0368,
        "G_d": 0.0435,
        "S_n": 0.0851,
        "S_alpha": 0.0901,
        "F_c": 0.1685,
    },
    "INSTA": {
        "O_alpha": 0.0779,
        "G_m": 0.0227,
        "G_d": 0.0280,
        "S_n": 0.0330,
        "S_alpha": 0.0270,
        "F_c": 0.2315,
    },
}

TABLE2_ABLATION = {
    "No PMV": {"O_alpha": 0.0625, "E_c": 0.1201, "F_c": 0.2412},
    "Naive PMV": {"O_alpha": 0.0649, "E_c": 0.1234, "F_c": 0.2474},
    "+HH matte": {"O_alpha": 0.0631, "E_c": 0.1242, "F_c": 0.2403},
    "+Rank/bin": {"O_alpha": 0.0646, "E_c": 0.1233, "F_c": 0.2477},
    "+Align": {"O_alpha": 0.0630, "E_c": 0.1233, "F_c": 0.2374},
    "PrismAvatar": {"O_alpha": 0.0607, "E_c": 0.1268, "F_c": 0.2300},
}

TABLE3_RUNTIME = {
    "live_tracker": {
        "fps": 10.65,
        "tracking_ms": 109.15,
        "render_32_views_ms": 18.59,
        "subpixel_ms": 1.67,
        "display_ms": 1.23,
    },
    "student_driver": {
        "fps": 38.49,
        "tracking_ms": 4.68,
        "render_32_views_ms": 16.09,
        "subpixel_ms": 1.67,
        "display_ms": 0.79,
    },
}


def benchmarks_bundle() -> dict:
    return {
        "table1_marcel": TABLE1_MARCEL,
        "table2_ablation": TABLE2_ABLATION,
        "table3_runtime": TABLE3_RUNTIME,
    }
