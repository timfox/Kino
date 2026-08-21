"""GrowLoop evaluation smoke (arXiv:2605.28882)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.growloop.config import GrowLoopConfig
from ltx_trainer.growloop.pipeline import pipeline_demo


def evaluation_smoke(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    c = cfg or GrowLoopConfig()
    demo = pipeline_demo(c, seed=42)
    h = demo["heuristic"]
    g = demo["gates"]
    runner = demo["dual_loop_runner"]
    return {
        "paper": c.paper_arxiv,
        "case_count": c.case_count,
        "merged_agreement_anchor": c.merged_agreement_gemini,
        "growloop_tie_aware": c.growloop_tie_aware_acc,
        "safety_converged": h["safety_converged"],
        "quality_converged": h["quality_converged"],
        "all_gates_pass": g["pass"],
        "dual_loop_actions": runner["actions_taken"],
        "dual_loop_agreement": runner["final_agreement"],
    }
