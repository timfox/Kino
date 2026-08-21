"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.era_defocus.paper import paper_card
from ltx_trainer.era_defocus.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    summary = demo["summary"]
    return {
        "paper": paper_card(),
        "demo": demo,
        "psnr_anchor": summary["DPDD_PSNR"],
        "status": "ok"
        if summary["DPDD_PSNR"] >= 26.6 and summary["gain_vs_best_baseline_db"] > 0
        else "fail",
    }
