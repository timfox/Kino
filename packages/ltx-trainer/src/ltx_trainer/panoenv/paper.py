"""Paper integration card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoenv.benchmarks import benchmarks_bundle, table3_ours
from ltx_trainer.panoenv.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    ref = table3_ours()
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
        "paradigm": "PanoEnv-RL (GRPO + routed GT rewards + two-stage curriculum)",
        "task": "3D spatial VQA on ERP panoramas (PanoEnv-QA)",
        "method": {
            "dataset": "14.8K QA from TartanAir ERP + 3D GT",
            "GRPO": "group-relative advantage, K=4",
            "rewards": "five strategies: yes/no, mcq, distance, spatial, counting",
            "curriculum": "Stage 1 structured (T/F, MCQ) → Stage 2 mixed OE",
            "base": "Qwen2.5-VL-7B-Instruct + LoRA decoder",
        },
        "results": {
            "total_acc": ref["total"],
            "oe_acc": ref["oe"],
            "q_score": 6.24,
            "p_score": 5.95,
            "vs_base_7b": "+3.59% total, OE 6.39% → 14.83%",
        },
        "reference_metrics": benchmarks_bundle(),
        "integration": (
            "GOPEX stubs PanoEnv-QA taxonomy, routed rewards, GRPO, and Table 2–4/7 anchors. "
            "Production: github.com/7zk1014/PanoEnv."
        ),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.panoenv.mock import evaluation_smoke

    return evaluation_smoke()
