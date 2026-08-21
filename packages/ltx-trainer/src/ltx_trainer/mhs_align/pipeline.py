"""Framework card and benchmarks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mhs_align.config import MhsAlignConfig
from ltx_trainer.mhs_align.layout import (
    BEHAVIORAL_ATTRIBUTES,
    EVALUATIVE_ATTRIBUTES,
    LIMITATIONS,
    MHS_ATTRIBUTES,
    PIPELINE_STAGES,
)
from ltx_trainer.mhs_align.mock import evaluation_smoke
from ltx_trainer.mhs_align.tables import (
    headline_results,
    table1_spearman_large,
    table2_ridge_reconstruction,
    table7_ablation,
)


def framework_card(cfg: MhsAlignConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MhsAlignConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "corpus": cfg.corpus,
        "corpus_scale": {
            "comments": cfg.corpus_comments,
            "annotators": cfg.corpus_annotators,
        },
        "problem": (
            "Holistic LLM hate-speech labels hide which subjective facets align or invert "
            "relative to human MHS annotators."
        ),
        "main_finding": cfg.main_finding,
        "attributes": list(MHS_ATTRIBUTES),
        "behavioral_cluster": list(BEHAVIORAL_ATTRIBUTES),
        "evaluative_cluster": list(EVALUATIVE_ATTRIBUTES),
        "models": list(cfg.models_evaluated),
        "prompting": ("vanilla", "persona_demographics"),
        "score_target": "IRT continuous hate speech score (θ_n)",
        "reconstruction": "confidence-weighted Ridge on per-attribute LLM ordinals",
        "pipeline_stages": list(PIPELINE_STAGES),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_spearman_large": table1_spearman_large(),
        "table2_ridge_reconstruction": table2_ridge_reconstruction(),
        "table7_ablation": table7_ablation(),
        "headlines": headline_results(),
    }


def evaluation_demo() -> dict[str, Any]:
    return evaluation_smoke()
