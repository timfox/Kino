"""Paper tables and human-study numbers (AniClipart + ChallengeSVG)."""

from __future__ import annotations

from typing import Any


def table_benchmark_structure() -> dict[str, dict[str, float]]:
    """Suppl. Table 3 — SVG structural statistics."""
    return {
        "AniClipart": {
            "num_svgs": 43,
            "elements_mean": 26.7,
            "elements_median": 23,
            "colors_mean": 6.5,
            "path_coords_mean": 1149.4,
        },
        "ChallengeSVG": {
            "num_svgs": 35,
            "elements_mean": 54.9,
            "elements_median": 45,
            "colors_mean": 17.5,
            "path_coords_mean": 1415.5,
        },
    }


def table_aniclipart_quantitative() -> dict[str, dict[str, float]]:
    """Table 2 — automatic metrics on AniClipart (paper values)."""
    return {
        "No Animation": {"XCLIP": 0.211, "LPIPS": 0.000, "SSIM": 1.000, "DOVER": 0.444, "Diversity": 0.000},
        "Vector Prism": {"XCLIP": 0.211, "LPIPS": 0.032, "SSIM": 0.973, "DOVER": 0.451, "Diversity": 0.012},
        "LiveSketch": {"XCLIP": 0.206, "LPIPS": 0.153, "SSIM": 0.910, "DOVER": 0.496, "Diversity": 0.026},
        "AniClipart": {"XCLIP": 0.214, "LPIPS": 0.104, "SSIM": 0.937, "DOVER": 0.427, "Diversity": 0.020},
        "FlexiClip": {"XCLIP": 0.213, "LPIPS": 0.092, "SSIM": 0.938, "DOVER": 0.431, "Diversity": 0.053},
        "LINR-Bridge": {"XCLIP": 0.215, "LPIPS": 0.174, "SSIM": 0.925, "DOVER": 0.433, "Diversity": 0.016},
        "LiveSVG (Veo 3.1)": {"XCLIP": 0.216, "LPIPS": 0.087, "SSIM": 0.942, "DOVER": 0.447, "Diversity": 0.052},
        "LiveSVG (LTX 2.3)": {"XCLIP": 0.215, "LPIPS": 0.105, "SSIM": 0.940, "DOVER": 0.445, "Diversity": 0.099},
        "LiveSVG (WAN 2.2)": {"XCLIP": 0.214, "LPIPS": 0.116, "SSIM": 0.938, "DOVER": 0.446, "Diversity": 0.063},
    }


def table_challengesvg_quantitative() -> dict[str, dict[str, float]]:
    """Suppl. Table 4 — ChallengeSVG metrics (WAN variant in paper)."""
    return {
        "No Animation": {"XCLIP": 0.214, "LPIPS": 0.000, "SSIM": 1.000, "DOVER": 0.470},
        "Vector Prism": {"XCLIP": 0.211, "LPIPS": 0.139, "SSIM": 0.867, "DOVER": 0.461},
        "LiveSketch": {"XCLIP": 0.182, "LPIPS": 0.503, "SSIM": 0.609, "DOVER": 0.397},
        "AniClipart": {"XCLIP": 0.204, "LPIPS": 0.274, "SSIM": 0.781, "DOVER": 0.433},
        "FlexiClip": {"XCLIP": 0.201, "LPIPS": 0.298, "SSIM": 0.773, "DOVER": 0.424},
        "LINR-Bridge": {"XCLIP": 0.205, "LPIPS": 0.491, "SSIM": 0.729, "DOVER": 0.426},
        "LiveSVG (WAN 2.2)": {"XCLIP": 0.215, "LPIPS": 0.208, "SSIM": 0.844, "DOVER": 0.476},
    }


def table_runtime_minutes() -> dict[str, float]:
    """Wall-clock minutes per SVG (paper + suppl. Table 7 subset)."""
    return {
        "Vector Prism": 2.9,
        "LiveSketch": 35.9,
        "AniClipart": 22.1,
        "FlexiClip": 57.4,
        "LINR-Bridge": 79.5,
        "LiveSVG (Veo 3.1)": 9.0,
        "LiveSVG (LTX 2.3)": 9.0,
        "LiveSVG (WAN 2.2)": 9.0,
    }


def human_preference_rates() -> dict[str, Any]:
    """Sec. 4.3 — overall human preference win rates (%)."""
    return {
        "AniClipart_overall": {
            "LiveSVG_aggregate": 86.7,
            "LiveSVG_Veo": 66.7,
            "LiveSVG_WAN": 64.3,
            "LiveSVG_LTX": 51.0,
            "vs_Vector_Prism": 76.2,
            "vs_LiveSketch": 100.0,
            "vs_AniClipart": 90.5,
            "vs_FlexiClip": 83.3,
            "vs_LINR_Bridge": 83.3,
        },
        "ChallengeSVG_overall_win_pct": 84.8,
        "ChallengeSVG_gemini_ab_win_pct": 77.8,
    }


def table_ablation_gemini() -> dict[str, str]:
    """Suppl. Table 6 — component ablations (overall LiveSVG wins)."""
    return {
        "w/o homography": "22/42 (52.4%)",
        "w/o tracking init": "28/42 (66.7%)",
        "w/o spatial reg": "30/42 (71.4%)",
        "w/o G1 reg": "22/42 (52.4%)",
    }
