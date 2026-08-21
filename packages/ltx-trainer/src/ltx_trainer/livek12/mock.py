"""LiveK12Bench evaluation smoke (arXiv:2605.26781)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livek12.config import Livek12Config
from ltx_trainer.livek12.scoring import arl, mock_exam_scoring_smoke
from ltx_trainer.livek12.tables import figure1_degradation, headline_results, table1_modality_counts


def evaluation_smoke(cfg: Livek12Config | None = None) -> dict[str, Any]:
    c = cfg or Livek12Config()
    ar = arl([1, 0, 1], [4000, 8000, 3500], c.avg_response_length_L_bar, c.efficiency_lambda)
    mock = mock_exam_scoring_smoke()
    fig1 = figure1_degradation()
    gpt5 = next(r for r in fig1 if r["model"] == "GPT-5")
    return {
        "paper": c.paper_arxiv,
        "disciplines": list(c.disciplines),
        "modalities": list(c.modalities),
        "mock_exam_scoring": mock,
        "arl_toy": round(ar, 4),
        "gpt5_oes_drop": round(gpt5["default"] - gpt5["exam"], 1),
        "table1_modalities": table1_modality_counts(),
        "headline": headline_results()["visual"],
    }
