"""Paper Tables 1–2 (Seshimo & Isogawa, arXiv:2509.00396)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.daovi.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

TABLE1_BASELINES = {
    "FuseFormer": {
        "psnr": 29.18,
        "ssim": 0.9727,
        "ws_psnr": 28.87,
        "ws_ssim": 0.9392,
        "vfid": 0.320,
    },
    "STTN": {
        "psnr": 32.21,
        "ssim": 0.9815,
        "ws_psnr": 31.09,
        "ws_ssim": 0.9520,
        "vfid": 0.238,
    },
    "ProPainter": {
        "psnr": 32.81,
        "ssim": 0.9877,
        "ws_psnr": 32.86,
        "ws_ssim": 0.9648,
        "vfid": 0.149,
    },
    "DAOVI": {
        "psnr": 33.37,
        "ssim": 0.9888,
        "ws_psnr": 33.37,
        "ws_ssim": 0.9661,
        "vfid": 0.138,
    },
}

TABLE2_ABLATION = {
    "w/o_GFCIP": {
        "psnr": 33.25,
        "ssim": 0.9884,
        "ws_psnr": 33.25,
        "ws_ssim": 0.9656,
        "vfid": 0.136,
    },
    "w/o_ODAFP": {
        "psnr": 32.84,
        "ssim": 0.9878,
        "ws_psnr": 32.89,
        "ws_ssim": 0.9650,
        "vfid": 0.148,
    },
    "DAOVI_full": {
        "psnr": 33.37,
        "ssim": 0.9888,
        "ws_psnr": 33.37,
        "ws_ssim": 0.9661,
        "vfid": 0.138,
    },
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1": TABLE1_BASELINES,
        "table2_ablation": TABLE2_ABLATION,
        "metrics": ("psnr", "ssim", "ws_psnr", "ws_ssim", "vfid"),
    }
