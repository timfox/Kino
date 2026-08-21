"""Agent-facing LiveK12Bench facts."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livek12.config import Livek12Config
from ltx_trainer.livek12.layout import CHALLENGING_SUBSETS, DISCIPLINES, MODALITIES


def livek12_knowledge_blob(cfg: Livek12Config | None = None) -> dict[str, Any]:
    cfg = cfg or Livek12Config()
    from ltx_trainer.livek12.tables import figure1_degradation, headline_results

    return {
        "name": "LiveK12Bench",
        "paper": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "scale": {
            "questions": cfg.n_questions,
            "knowledge_points": cfg.n_knowledge_points,
            "disciplines": list(DISCIPLINES),
        },
        "modalities": [m[0] for m in MODALITIES],
        "challenging_subsets": [s[0] for s in CHALLENGING_SUBSETS],
        "mock_exam_dims": ["outcome_accuracy", "process_quality", "reasoning_efficiency", "OES"],
        "figure1": figure1_degradation(),
        "headlines": headline_results(),
        "gopex_tools": [
            "livek12_knowledge",
            "livek12_framework_card",
            "livek12_eval_demo",
            "livek12_benchmarks",
            "livek12_mock_exam_score",
        ],
        "related_gopex": {
            "mhs_align": "MHS hate-speech attribute alignment (arXiv:2605.27025)",
            "cvsearch": "Cognitive visual search for HR MLLMs",
        },
    }
