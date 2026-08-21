"""Paper table excerpts (Tables 3–7, Figs 4/9)."""

from __future__ import annotations

from typing import Any


def table_t2av_main_results() -> list[dict[str, Any]]:
    """Table 3 — T2AV main results (excerpt from paper)."""
    return [
        _row_t2av("Seedance 2.0", True, 0.9023, 3.7116, 4.2649, 4.0065, 4.1128, 0.6183, 3.6038, 3.7875, 4.1845),
        _row_t2av("Kling 3.0", True, 0.9274, 3.3893, 4.4139, 3.8502, 3.8542, 0.6185, 3.4922, 3.6049, 3.7713),
        _row_t2av("Veo 3.1", True, 0.7784, 2.8961, 3.1348, 4.0032, 3.5759, 0.6142, 3.3490, 3.2387, 3.6931),
        _row_t2av("LTX 2.3", True, 0.7321, 2.2880, 3.2888, 3.8829, 3.0203, 0.6205, 2.7278, 2.5017, 2.9313),
        _row_t2av("Wan2.2-I2V-A14B", False, 0.5994, 2.0046, 2.2576, 3.5747, 2.6794, 0.6123, None, None, None),
        _row_t2av("Open-Sora", False, 0.2476, 1.3854, 1.4947, 3.6418, 1.5676, 0.6161, None, None, None),
        _row_t2av("VideoDirectorGPT", False, 0.5205, 2.0990, 1.8172, 3.3830, 2.4549, 0.6155, None, None, None),
    ]


def _row_t2av(
    model: str,
    aud: bool,
    vqa: float,
    vq: float,
    cont: float,
    trans: float,
    hol: float,
    tva: float,
    avs: float | None,
    audq: float | None,
    audl: float | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "audio": aud,
        "VQA": vqa,
        "VQ": vq,
        "Cont.": cont,
        "Trans.": trans,
        "Hol.": hol,
        "TVAlign": tva,
        "AVS": avs,
        "AudQ": audq,
        "AudL": audl,
    }


def table_i2av_main_results() -> list[dict[str, Any]]:
    """Table 4 — I2AV main results (excerpt)."""
    return [
        {
            "model": "Seedance 2.0",
            "VQA": 0.9204,
            "VQ": 3.7651,
            "Cont.": 4.9182,
            "IV1": 0.9622,
            "ImgAlign": 0.9027,
            "AVS": 3.5669,
        },
        {
            "model": "Kling 3.0",
            "VQA": 0.8939,
            "VQ": 3.2760,
            "Cont.": 4.1244,
            "IV1": 0.9960,
            "ImgAlign": 0.8877,
        },
        {
            "model": "VideoDirectorGPT",
            "VQA": 0.1976,
            "VQ": 1.5073,
            "Cont.": 1.0000,
            "IV1": 0.9303,
            "ImgAlign": 0.9640,
        },
    ]


def table_v2av_main_results() -> list[dict[str, Any]]:
    """Table 5 — V2AV main results."""
    return [
        {"model": "Seedance 2.0", "VQA": 0.8753, "VQ": 3.8336, "Cont.": 4.7636, "TVAlign": 0.9727, "AVS": 3.7591},
        {"model": "Veo 3.1", "VQA": 0.8055, "VQ": 3.0869, "Cont.": 1.8425, "Trans.": 2.2815, "TVAlign": 0.7100},
        {"model": "Helios (14B)", "VQA": 0.4818, "VQ": 1.8197, "Cont.": 2.0324, "audio": False},
    ]


def table_difficulty_balanced_scores() -> list[dict[str, Any]]:
    """Table 6 — per-difficulty balanced scores."""
    return [
        {"family": "Proprietary Models", "L1": 70.6, "L2": 75.2, "L3": 74.5, "L4": 73.9},
        {"family": "Open-Source Models", "L1": 57.9, "L2": 52.9, "L3": 52.8, "L4": 51.4},
        {"family": "Agent-Based Models", "L1": 47.3, "L2": 47.4, "L3": 43.2, "L4": 41.2},
    ]


def table_scenario_balanced_scores_t2av() -> dict[str, dict[str, float]]:
    """Fig. 4 excerpt — scenario-level balanced scores (T2AV)."""
    return {
        "Performance Ads": {"Seedance 2.0": 82.1, "Kling 3.0": 80.4, "LTX 2.3": 58.2, "Open-Sora": 41.0},
        "Content-Creator": {"Seedance 2.0": 84.0, "Kling 3.0": 82.5, "LTX 2.3": 59.1, "Open-Sora": 43.5},
        "Brand Ads": {"Seedance 2.0": 83.2, "Kling 3.0": 81.8, "LTX 2.3": 57.4, "Open-Sora": 42.2},
        "Vlog": {"Seedance 2.0": 81.5, "Kling 3.0": 79.9, "LTX 2.3": 56.8, "Open-Sora": 40.5},
    }


def table_event_count_balanced_scores() -> dict[str, dict[str, float]]:
    """Fig. 9 excerpt — short vs long event chains."""
    return {
        "short_chains_lte_4": {
            "Proprietary Models": 75.8,
            "Open-Source Models": 58.6,
            "Agent-Based Models": 46.8,
        },
        "long_chains_gt_4": {
            "Proprietary Models": 74.1,
            "Open-Source Models": 52.4,
            "Agent-Based Models": 44.9,
        },
    }


def table_input_format_sensitivity() -> list[dict[str, Any]]:
    """Table 7 — same content under V2AV / I2AV / T2AV conditioning."""
    return [
        {"model": "Seedance 2.0", "V2AV": 80.4, "I2AV": 83.9, "T2AV": 83.6},
        {"model": "Veo 3.1", "V2AV": 57.4, "I2AV": 71.8, "T2AV": 68.1},
        {"model": "LongCat", "V2AV": 39.8, "I2AV": 40.4, "T2AV": 41.2},
        {"model": "Helios (14B)", "V2AV": 40.5, "I2AV": 34.4, "T2AV": 34.6},
    ]


def evaluated_models_list() -> list[dict[str, str]]:
    """Sec. 4.1 — 11 systems under evaluation."""
    return [
        {"name": "Seedance 2.0", "category": "proprietary"},
        {"name": "Kling 3.0", "category": "proprietary"},
        {"name": "Veo 3.1", "category": "proprietary"},
        {"name": "LTX 2.3", "category": "open-source"},
        {"name": "LongCat", "category": "open-source"},
        {"name": "Wan2.2-I2V-A14B", "category": "open-source"},
        {"name": "HunyuanVideo 1.5-I2V", "category": "open-source"},
        {"name": "Helios (14B)", "category": "open-source"},
        {"name": "Open-Sora", "category": "open-source"},
        {"name": "daVinci-MagiHuman", "category": "open-source"},
        {"name": "VideoDirectorGPT", "category": "agent-based"},
    ]
