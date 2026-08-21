"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sa_music.benchmark import total_questions
from ltx_trainer.sa_music.config import SaMusicConfig
from ltx_trainer.sa_music.pipeline import benchmarks_bundle, evaluation_demo, table_understanding_anchors


def evaluation_smoke(cfg: SaMusicConfig | None = None) -> dict[str, Any]:
    c = cfg or SaMusicConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    assert c.benchmark_questions == 504
    assert total_questions() == 504
    assert c.models_evaluated == 33
    assert c.gemini_theory_acc == 0.908
    assert c.gemini_style_accuracy == 0.40
    assert c.gemini_genre_accuracy == 0.95
    assert demo["genre_exceeds_style"] is True
    assert demo["gemini_beats_opensource"] is True
    assert demo["kl_style_correlation_negative"] is True
    assert c.kl_style_pearson_r == -0.73

    gemini = next(r for r in table_understanding_anchors(c) if "Gemini" in r["model"])
    assert gemini["continuation"] == 0.852
    assert len(benchmarks_bundle(c)["telr_levels"]) == 5

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "benchmark_questions": c.benchmark_questions,
        "gemini_theory_acc": c.gemini_theory_acc,
        "gemini_style_accuracy": c.gemini_style_accuracy,
        "reference_scores": c.reference_abc_scores,
    }
