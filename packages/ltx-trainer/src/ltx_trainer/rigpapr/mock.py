"""Runnable evaluation smoke for RigPAPR (arXiv:2606.06685)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rigpapr.benchmarks import benchmarks_bundle, table1_ours
from ltx_trainer.rigpapr.config import PAPER_ARXIV
from ltx_trainer.rigpapr.pipeline import evaluation_demo


def evaluation_smoke() -> dict[str, Any]:
    ours = table1_ours()
    demo = evaluation_demo(seed=0)
    return {
        "package": "rigpapr",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_train_psnr": ours["synth_train_psnr"],
        "ref_novel_psnr": ours["synth_novel_psnr"],
        "novel_psnr_margin_db": demo["novel_gain_db"],
        "phase1_total": demo["phase1_losses"]["total"],
        "phase2_total": demo["phase2_losses"]["total"],
        "render_delta": demo["render_delta"],
    }
