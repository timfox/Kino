"""Paper tables and anchors (Peng et al. arXiv:2606.06685)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rigpapr.config import (
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    REAL_RES,
    SCENE_FRAMES,
    SYNTH_RES,
)

# Table 1 — synthetic + real (Sec. 4.2)
TABLE1_MAIN: list[dict[str, Any]] = [
    {
        "method": "Puppeteer",
        "synth_train_psnr": 22.48,
        "synth_train_ssim": 0.941,
        "synth_train_lpips": 0.044,
        "synth_novel_psnr": 19.27,
        "synth_novel_ssim": 0.923,
        "synth_novel_lpips": 0.075,
        "real_train_psnr": None,
    },
    {
        "method": "Mani-GS",
        "synth_train_psnr": 28.53,
        "synth_train_ssim": 0.974,
        "synth_train_lpips": 0.026,
        "synth_novel_psnr": 20.51,
        "synth_novel_ssim": 0.927,
        "synth_novel_lpips": 0.072,
        "real_train_psnr": 25.68,
        "real_train_ssim": 0.955,
        "real_train_lpips": 0.049,
    },
    {
        "method": "Ours",
        "synth_train_psnr": 29.07,
        "synth_train_ssim": 0.972,
        "synth_train_lpips": 0.027,
        "synth_novel_psnr": 23.63,
        "synth_novel_ssim": 0.945,
        "synth_novel_lpips": 0.044,
        "real_train_psnr": 25.82,
        "real_train_ssim": 0.944,
        "real_train_lpips": 0.063,
    },
]

# Table 2 — two-phase ablation synthetic (Sec. 4.4)
TABLE2_TWO_PHASE: list[dict[str, Any]] = [
    {"config": "Mani-GS (Phase 1)", "train_psnr": 26.71, "novel_psnr": 20.14},
    {"config": "Mani-GS (Full)", "train_psnr": 28.53, "novel_psnr": 20.51},
    {"config": "Ours (Phase 1)", "train_psnr": 28.13, "novel_psnr": 23.84},
    {"config": "Ours (Full)", "train_psnr": 29.07, "novel_psnr": 23.63},
]

# Table 3 — T-Rex optimization ablation (App. B.1)
TABLE3_TREX_ABLATION: list[dict[str, Any]] = [
    {
        "config": "Phase 1 w/o L∆depth",
        "train_psnr": 29.21,
        "novel_psnr": 21.38,
        "chamfer_x1e3": 15.64,
    },
    {"config": "Phase 1", "train_psnr": 27.72, "novel_psnr": 24.05, "chamfer_x1e3": 5.68},
    {"config": "Full", "train_psnr": 31.08, "novel_psnr": 23.95, "chamfer_x1e3": 4.60},
]

# Table 4 — scene frame counts (App. C.1)
_SYNTH_SCENES = {"t_rex", "simpsons", "fox", "wolf", "spider"}
TABLE4_SCENE_FRAMES: list[dict[str, Any]] = [
    {
        "scene": k,
        "frames": v,
        "resolution": f"{SYNTH_RES[0]}×{SYNTH_RES[1]}"
        if k in _SYNTH_SCENES
        else f"{REAL_RES[0]}×{REAL_RES[1]}",
    }
    for k, v in SCENE_FRAMES.items()
]

PAPER_ANCHORS = {
    "novel_psnr_gain_vs_mani_gs_db": 3.12,
    "train_psnr_ours": 29.07,
    "novel_psnr_ours": 23.63,
    "p_react_note": "fixed-viewpoint I2V (Kling 3.0) driving on real scenes",
}


def table1_ours() -> dict[str, Any]:
    return next(r for r in TABLE1_MAIN if r["method"] == "Ours")


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "venue": PAPER_VENUE,
            "url": PAPER_URL,
        },
        "table1_main": TABLE1_MAIN,
        "table2_two_phase": TABLE2_TWO_PHASE,
        "table3_trex_ablation": TABLE3_TREX_ABLATION,
        "table4_scene_frames": TABLE4_SCENE_FRAMES,
        "anchors": PAPER_ANCHORS,
    }
