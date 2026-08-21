"""Reference metrics from NTIRE 2026 RAIM Track 2 report (arXiv:2604.09030)."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV = "2604.09030"
PAPER_TITLE = "NTIRE 2026 RAIM Track 2: Multi-Exposure Image Fusion in Dynamic Scenes"
REPO_URL = "https://github.com/qulishen/RAIM-HDR"

DATASET = {
    "train_sequences": 100,
    "train_exposures_per_sequence": 7,
    "test_sequences": 100,
    "test_exposures_per_sequence": 5,
    "participating_teams": 114,
    "submissions": 987,
}

# Table 1 — verified final results (Test Stage 1 + Stage 2 average)
TABLE1_FINAL_RESULTS: dict[str, dict[str, float]] = {
    "WHU-VIP": {
        "psnr_s1": 27.059,
        "ssim_s1": 0.915,
        "lpips_s1": 0.089,
        "score_s1": 58.249,
        "psnr_s2": 27.556,
        "ssim_s2": 0.923,
        "lpips_s2": 0.080,
        "score_s2": 59.529,
        "final_score": 58.889,
    },
    "SHL": {
        "psnr_s1": 27.653,
        "ssim_s1": 0.920,
        "lpips_s1": 0.112,
        "score_s1": 57.110,
        "psnr_s2": 28.825,
        "ssim_s2": 0.928,
        "lpips_s2": 0.101,
        "score_s2": 58.958,
        "final_score": 58.034,
    },
    "nunucccb": {
        "psnr_s1": 26.839,
        "ssim_s1": 0.913,
        "lpips_s1": 0.111,
        "score_s1": 56.395,
        "psnr_s2": 27.011,
        "ssim_s2": 0.918,
        "lpips_s2": 0.099,
        "score_s2": 57.605,
        "final_score": 57.000,
    },
    "untrafusion": {
        "psnr_s1": 26.847,
        "ssim_s1": 0.908,
        "lpips_s1": 0.095,
        "score_s1": 57.333,
        "psnr_s2": 26.002,
        "ssim_s2": 0.907,
        "lpips_s2": 0.122,
        "score_s2": 54.733,
        "final_score": 56.033,
    },
    "I2_Group_Transsion": {
        "psnr_s1": 26.221,
        "ssim_s1": 0.904,
        "lpips_s1": 0.121,
        "score_s1": 54.853,
        "psnr_s2": 26.137,
        "ssim_s2": 0.909,
        "lpips_s2": 0.113,
        "score_s2": 55.619,
        "final_score": 55.236,
    },
    "miketjc": {
        "psnr_s1": 26.002,
        "ssim_s1": 0.907,
        "lpips_s1": 0.122,
        "score_s1": 54.733,
        "psnr_s2": 26.312,
        "ssim_s2": 0.909,
        "lpips_s2": 0.118,
        "score_s2": 55.343,
        "final_score": 55.038,
    },
}

EVALUATION = {
    "metrics": ("PSNR", "SSIM", "LPIPS", "DISTS", "NIQE"),
    "leaderboard_formula": (
        "Score = 30·PSNR/50 + 22.5·(SSIM−0.5)/0.5 + 30·(1−LPIPS/0.4)"
    ),
    "phase2_format": "001.jpg–100.jpg + readme.txt in one zip",
    "final_requires": ("inference code", "weights", "test.sh"),
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "table1_final_results": TABLE1_FINAL_RESULTS,
        "evaluation": EVALUATION,
        "repo": REPO_URL,
    }
