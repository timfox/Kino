"""Framework metadata."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hypevpr.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL
from ltx_trainer.hypevpr.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "components": {
            "problem": "Perspective-to-equirectangular (P2E) visual place recognition",
            "ham": "Hierarchical Aggregation Module in Poincaré ball (Einstein midpoint, Eq. 16–17)",
            "query_path": "GeM + expmap0 → h_q (Eq. 7–9)",
            "retrieval": "Coarse top-K' on h^(1,1) then multi-level z-score rerank (Eq. 18–20)",
            "losses": "L_hier + L_hyp + L_euc (Eq. 21–24)",
            "variants": "HypeVPR-O (level 1), -B (1+4), -L (1+5), -SW (level 5 only)",
        },
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "hypevpr", **evaluation_demo_run()}
