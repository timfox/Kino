"""MHS alignment evaluation smoke (arXiv:2605.27025)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mhs_align.config import MhsAlignConfig
from ltx_trainer.mhs_align.layout import BEHAVIORAL_ATTRIBUTES, EVALUATIVE_ATTRIBUTES
from ltx_trainer.mhs_align.ridge import ridge_reconstruction_smoke, token_confidence
from ltx_trainer.mhs_align.tables import attribute_clusters_from_table1, headline_results


def spearman_smoke() -> dict[str, Any]:
    clusters = attribute_clusters_from_table1()
    behavioral = [a for a, c in clusters.items() if c == "behavioral"]
    evaluative = [a for a, c in clusters.items() if c == "evaluative"]
    return {
        "behavioral_from_table1": behavioral,
        "evaluative_from_table1": evaluative,
        "matches_layout_behavioral": set(behavioral) == set(BEHAVIORAL_ATTRIBUTES),
        "matches_layout_evaluative": set(evaluative) == set(EVALUATIVE_ATTRIBUTES),
    }


def evaluation_smoke(cfg: MhsAlignConfig | None = None) -> dict[str, Any]:
    c = cfg or MhsAlignConfig()
    conf = token_confidence({0: -0.1, 1: -0.5, 2: 0.0, 3: -1.2, 4: -2.0}, chosen=2)
    ridge = ridge_reconstruction_smoke()
    spearman = spearman_smoke()
    return {
        "paper": c.paper_arxiv,
        "mhs_attributes": len(BEHAVIORAL_ATTRIBUTES) + len(EVALUATIVE_ATTRIBUTES),
        "token_confidence_example": round(conf, 4),
        "spearman_cluster_smoke": spearman,
        "ridge_reconstruction_smoke": ridge,
        "best_r2_paper": c.best_r2_large[0],
        "headline": headline_results()["reconstruction"],
    }
