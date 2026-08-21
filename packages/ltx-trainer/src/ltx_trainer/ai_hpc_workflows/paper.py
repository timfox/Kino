"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL
from ltx_trainer.ai_hpc_workflows.constants import AUTHOR, AFFILIATION, KEYWORDS, TIPS
from ltx_trainer.ai_hpc_workflows.orchestration import architecture_card, recommend_orchestrator
from ltx_trainer.ai_hpc_workflows.references import reference_anchors
from ltx_trainer.ai_hpc_workflows.tips import tips_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "author": AUTHOR,
        "affiliation": AFFILIATION,
        "keywords": list(KEYWORDS),
        "n_tips": len(TIPS),
        "prior_guide": "Fifteen quick tips for success with HPC (Alnasir 2021, PLoS Comp Bio)",
        "domain_emphasis": "computational biology: simulation, imaging, omics, iterative discovery",
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "tips": tips_card(),
        "architecture": architecture_card(),
        "orchestrator_static": recommend_orchestrator("static"),
        "orchestrator_dynamic": recommend_orchestrator("dynamic"),
        "references": reference_anchors(),
    }
