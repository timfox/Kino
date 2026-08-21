"""MUSTBENCH evaluation smoke (arXiv:2605.29300)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mustbench.config import MustBenchConfig
from ltx_trainer.mustbench.pipeline import pipeline_demo


def evaluation_smoke(cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    c = cfg or MustBenchConfig()
    demo = pipeline_demo(c, seed=42)
    return {
        "paper": c.paper_arxiv,
        "num_tasks": 5,
        "test_qa_pairs": c.test_qa_total,
        "must_7b_total": c.must_7b_total,
        "grpo_ordering_ok": demo["grpo_ordering_ok"],
        "computed_tsg_hit3": demo["computed_tsg_hit3"],
        "computed_gto_acc": demo["computed_gto_acc"],
        "must_tokens": demo["must_tokens"],
        "grpo_loss_finite": demo["grpo_loss_finite"],
        "grpo_reward_improved": demo["grpo_reward_improved"],
    }
