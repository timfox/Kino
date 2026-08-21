"""COMET evaluation smoke (arXiv:2605.29628)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.comet.config import CometConfig
from ltx_trainer.comet.pipeline import pipeline_demo


def evaluation_smoke(cfg: CometConfig | None = None) -> dict[str, Any]:
    c = cfg or CometConfig()
    demo = pipeline_demo(c, seed=0)
    return {
        "paper": c.paper_arxiv,
        "embed_dim": c.embed_dim,
        "head_size": c.head_size,
        "sigma_top3": demo["sigma_top3"],
        "plshead_dims": demo["plshead_dims"],
        "similarity_direct": demo["similarity_direct"],
        "retrieval_R1": demo["retrieval_R1"],
    }
