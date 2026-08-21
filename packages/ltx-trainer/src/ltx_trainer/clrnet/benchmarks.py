"""Reference metrics from Kegl et al. (arXiv:2603.15767)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clrnet.config import (
    DUAL_RADAR_FRAMES,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    CODE_URL,
    VOD_TEST_FRAMES,
    VOD_TRAIN_FRAMES,
)

# Table II — VoD camera-radar MAE (±20 cm, ±1°)
TABLE2_ABLATION: list[dict[str, Any]] = [
    {"config": "CRNet vanilla", "transl_cm": 20.0, "rot_deg": 0.8},
    {"config": "CRNet + equirectangular", "transl_cm": 14.0, "rot_deg": 0.6},
    {"config": "CRNet + 5-frame radar", "transl_cm": 18.2, "rot_deg": 0.7},
    {"config": "CRNet + extra radar channels", "transl_cm": 16.6, "rot_deg": 0.6},
    {"config": "CRNet + camera depth branch", "transl_cm": 18.8, "rot_deg": 0.9},
    {"config": "CRNet (all features)", "transl_cm": 11.3, "rot_deg": 0.6},
    {"config": "CLRNet + loop closure", "transl_cm": 9.5, "rot_deg": 0.5},
    {"config": "CLRNet + shared feature space", "transl_cm": 9.1, "rot_deg": 0.4},
    {"config": "Proposed CLRNet (all features)", "transl_cm": 9.1, "rot_deg": 0.4},
]

# Table III — selected rows (median camera-radar / camera-lidar)
TABLE3_VOD: list[dict[str, Any]] = [
    {
        "scenario": "non-rigid one model [±20cm, ±1°]",
        "method": "4DRC-OC",
        "cr_transl_median": 18.6,
        "cr_rot_median": 0.8,
        "cl_transl_median": None,
        "cl_rot_median": None,
    },
    {
        "scenario": "non-rigid one model [±20cm, ±1°]",
        "method": "CRNet (ours)",
        "cr_transl_median": 10.0,
        "cr_rot_median": 0.5,
        "cl_transl_median": None,
        "cl_rot_median": None,
    },
    {
        "scenario": "non-rigid one model [±20cm, ±1°]",
        "method": "CLRNet (ours)",
        "cr_transl_median": 7.8,
        "cr_rot_median": 0.4,
        "cl_transl_median": 3.7,
        "cl_rot_median": 0.1,
    },
    {
        "scenario": "rigid iterative [±100cm, ±20°]",
        "method": "CLRNet+4 (ours)",
        "cr_transl_median": 0.9,
        "cr_rot_median": 0.1,
        "cl_transl_median": 0.3,
        "cl_rot_median": 0.03,
    },
    {
        "scenario": "rigid iterative [±200cm, ±180°]",
        "method": "CLRNet+4 (ours)",
        "cr_transl_median": 1.4,
        "cr_rot_median": 0.2,
        "cl_transl_median": 0.2,
        "cl_rot_median": 0.04,
    },
]

# Table IV — domain transfer (median camera-radar)
TABLE4_DOMAIN: list[dict[str, Any]] = [
    {"method": "4DRC-OC", "train": "VoD", "test": "Dual", "transl_cm": 186, "rot_deg": 8.3},
    {"method": "4DRC-OC", "train": "Dual", "test": "Dual", "transl_cm": 3.8, "rot_deg": 0.2},
    {"method": "CRNet (ours)", "train": "VoD", "test": "Dual", "transl_cm": 139, "rot_deg": 6.3},
    {"method": "CRNet (ours)", "train": "Dual", "test": "Dual", "transl_cm": 1.8, "rot_deg": 0.2},
    {"method": "CRNet (ours)", "train": "VoD", "test": "VoD", "transl_cm": 1.8, "rot_deg": 0.2},
]


def table2_clrnet_full() -> dict[str, Any]:
    return next(r for r in TABLE2_ABLATION if "Proposed CLRNet" in r["config"])


def table3_clrnet_median() -> dict[str, Any]:
    return next(r for r in TABLE3_VOD if r["method"] == "CLRNet (ours)")


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "datasets": {
            "VoD": {"train": VOD_TRAIN_FRAMES, "test": VOD_TEST_FRAMES},
            "Dual-Radar": {"frames": DUAL_RADAR_FRAMES},
        },
        "table2_ablation": TABLE2_ABLATION,
        "table3_vod": TABLE3_VOD,
        "table4_domain": TABLE4_DOMAIN,
    }
