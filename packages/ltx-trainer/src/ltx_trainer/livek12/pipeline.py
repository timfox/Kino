"""Framework card and benchmarks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livek12.config import Livek12Config
from ltx_trainer.livek12.layout import (
    CHALLENGING_SUBSETS,
    DISCIPLINES,
    LIMITATIONS,
    MODALITIES,
    PIPELINE_STAGES,
    PROCESS_ERROR_TYPES,
    QUESTION_TYPES,
)
from ltx_trainer.livek12.mock import evaluation_smoke
from ltx_trainer.livek12.tables import (
    figure1_degradation,
    headline_results,
    table3_gemini_math_row,
    table3_gpt5_biology_row,
    table4_complex_layout_io_drop,
)


def framework_card(cfg: Livek12Config | None = None) -> dict[str, Any]:
    cfg = cfg or Livek12Config()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Static K-12 benchmarks leak into training data, use single-score accuracy, and "
            "rely on human OCR/cropping — mis-estimating real exam readiness."
        ),
        "proposal": (
            "LiveK12Bench: dynamic real-exam ingestion + Mock Exam protocol (outcome, process, "
            "efficiency, weighted OES) across TO/TI/IO modalities."
        ),
        "dataset": {
            "questions": cfg.n_questions,
            "knowledge_points": cfg.n_knowledge_points,
            "disciplines": list(DISCIPLINES),
            "question_types": list(QUESTION_TYPES),
            "modalities": [{"id": m[0], "desc": m[1]} for m in MODALITIES],
            "subsets": [{"id": s[0], "desc": s[1]} for s in CHALLENGING_SUBSETS],
        },
        "process_errors": list(PROCESS_ERROR_TYPES),
        "hyperparameters": {
            "tau": cfg.process_penalty_tau,
            "lambda_arl": cfg.efficiency_lambda,
            "wp": cfg.process_weight_wp,
            "L_bar": cfg.avg_response_length_L_bar,
        },
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "figure1_degradation": figure1_degradation(),
        "table3_gemini_math": table3_gemini_math_row(),
        "table3_gpt5_biology": table3_gpt5_biology_row(),
        "table4_io_layout_drop": table4_complex_layout_io_drop(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return evaluation_smoke()


def mock_exam_score_demo(
    point_values: list[float] | None = None,
    outcome_points: list[float] | None = None,
    cie: int = 0,
    lae: int = 0,
    dre: int = 0,
) -> dict[str, Any]:
    from ltx_trainer.livek12.scoring import exam_score_item, overall_exam_score, process_score

    cfg = Livek12Config()
    pvs = point_values or [10.0, 12.0]
    outs = outcome_points or [10.0, 8.0]
    errors = {"CIE": cie, "LAE": lae, "DRE": dre}
    rows = []
    for v, o in zip(pvs, outs, strict=True):
        p = process_score(v, errors, cfg.process_penalty_tau)
        es = exam_score_item(v, o, p, cfg.process_weight_wp)
        rows.append({"V": v, "O": o, "P": round(p, 2), "ES": round(es, 2)})
    return {
        "items": rows,
        "OES": round(overall_exam_score([r["ES"] for r in rows], pvs), 2),
        "errors": errors,
    }
