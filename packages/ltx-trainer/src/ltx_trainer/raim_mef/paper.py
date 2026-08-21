"""NTIRE 2026 RAIM Track 2 paper stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.raim_mef.benchmarks import PAPER_ARXIV, PAPER_TITLE, REPO_URL, benchmarks_bundle


def framework_card() -> dict[str, Any]:
    b = benchmarks_bundle()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "organizers": "Qu, Liu, Liang, Zeng, Dai (OPPO / Nankai / PolyU / Würzburg)",
        "problem": (
            "Fuse exposure brackets in dynamic scenes with motion, illumination change, and shake "
            "without ghosting; sRGB display-ready MEF (not linear HDR radiance)."
        ),
        "dataset": b["dataset"],
        "evaluation": b["evaluation"],
        "winner": "WHU-VIP (AFUNet + SFT + DCM-RG, 58.889 final score)",
        "repo": REPO_URL,
        "reference_metrics": b,
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.raim_mef.mock import evaluation_smoke

    return evaluation_smoke()
