"""MER-with-LLMs survey smoke (arXiv:2605.21239)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mer_llm.config import MerLlmConfig
from ltx_trainer.mer_llm.challenges import three_challenges
from ltx_trainer.mer_llm.scoring import explainable_mer_stub, paradigm_lift
from ltx_trainer.mer_llm.tables import fig1b_paradigm_progress
from ltx_trainer.mer_llm.taxonomy import five_subtasks, taxonomy_branches


def evaluation_smoke(cfg: MerLlmConfig | None = None) -> dict[str, Any]:
    c = cfg or MerLlmConfig()
    gvec = next(r for r in fig1b_paradigm_progress() if r["sub_task"] == "GVEC")
    lift = paradigm_lift(
        gvec["small_scale_sota"],
        gvec["general_mllm_zero_shot"],
        gvec["mer_with_llms"],
    )
    demo = explainable_mer_stub(
        observation="Relaxed expression and steady speech suggest positive affect.",
        emotion="happy",
    )
    return {
        "paper": c.paper_arxiv,
        "subtask_count": len(c.subtasks),
        "branch_count": len(taxonomy_branches()),
        "challenge_count": len(three_challenges()),
        "gvec_lift_vs_mllm": lift["vs_mllm_zero_shot"],
        "emoverse_emoset_acc": c.emoverse_emoset_acc,
        "demo_has_thinking": "thinking" in demo,
        "subtasks": [s["id"] for s in five_subtasks()],
    }
