"""Ethical pluralism evaluation smoke (arXiv:2605.28707)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ethical_pluralism.config import EthicalPluralismConfig
from ltx_trainer.ethical_pluralism.pipeline import evaluation_demo
from ltx_trainer.ethical_pluralism.simplex import project_simplex, simplex_constraint_ok


def evaluation_smoke(cfg: EthicalPluralismConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EthicalPluralismConfig()
    demo = evaluation_demo(seed=0)
    scores = project_simplex(0.5, 0.3, 0.2)
    return {
        "paper": "arXiv:2605.28707",
        "simplex_ok": simplex_constraint_ok(scores),
        "cv_em_accuracy": demo["cv_metrics"]["em_accuracy"],
        "test_em_accuracy": demo["test_metrics"]["em_accuracy"],
        "test_macro_f1": demo["test_metrics"]["macro_f1"],
        "paper_em_accuracy": cfg.paper_em_accuracy,
        "n_subtheories": 15,
        "benchmark_cases": 450,
        "ablation_rows": len(demo["ablation"]),
    }
