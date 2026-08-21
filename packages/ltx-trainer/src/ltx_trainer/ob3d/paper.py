"""Framework metadata."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ob3d.config import GITHUB_URL, KAGGLE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.ob3d.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "kaggle": KAGGLE_URL,
        "github": GITHUB_URL,
        "components": {
            "dataset": "12 Blender scenes × ego/non-ego × 100 ERP views @ 1600×800",
            "ground_truth": "RGB, depth, normals, exact Blender cameras, OpenMVG sparse points",
            "cpe": "RRA, RTA, AUC@5, ATE on relative poses",
            "nvs": "PSNR, SSIM, LPIPS (A/V) on held-out views",
            "recon": "Mesh-rendered depth vs GT; RMSE, MAE, AbsRel, δ1.25 (sky masked)",
        },
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "ob3d", **evaluation_demo_run()}
