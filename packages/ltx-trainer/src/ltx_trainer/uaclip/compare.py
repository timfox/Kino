"""Compare synthetic candidates against paper baseline tables (Tables 2, 5)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.uaclip.baselines import TABLE2_AMAZON_EDITING, TABLE5_AIRBNB_EDITING
from ltx_trainer.uaclip.demand import Platform, airbnb_demand_score, amazon_demand_score
from ltx_trainer.uaclip.generator import CandidateImage, score_candidates
from ltx_trainer.uaclip.similarity import l2_normalize
from ltx_trainer.uaclip.visual_attrs import VisualAttributes


def _table_for(platform: Platform) -> list[dict[str, Any]]:
    return TABLE2_AMAZON_EDITING if platform == "amazon" else TABLE5_AIRBNB_EDITING


def score_candidate_batch(
    platform: Platform,
    *,
    seed: int = 0,
    n_candidates: int = 8,
    embed_dim: int = 64,
) -> dict[str, Any]:
    """Rank random candidates; report demand vs paper Utility-Aware Generator row."""
    rng = np.random.default_rng(seed)
    text_emb = l2_normalize(rng.standard_normal(embed_dim))
    ref_emb = l2_normalize(rng.standard_normal(embed_dim))
    cands = [
        CandidateImage(
            f"c{i}",
            l2_normalize(rng.standard_normal(embed_dim)),
            VisualAttributes.random(rng, platform),
        )
        for i in range(n_candidates)
    ]
    ranked = score_candidates(cands, text_emb, ref_emb, platform)
    best = ranked[0]
    paper_uag = next(r for r in _table_for(platform) if r["model"] == "Utility-Aware Generator")
    return {
        "platform": platform,
        "best_candidate": best.to_dict(),
        "paper_uag_demand": paper_uag["demand"],
        "beats_paper_uag_demand": best.demand_score >= float(paper_uag["demand"]),
        "ranked_top3": [c.to_dict() for c in ranked[:3]],
    }


def demand_at_optimal(platform: Platform) -> dict[str, float]:
    """Demand score at paper inverted-U peaks (Amazon) or moderate Airbnb attrs."""
    if platform == "amazon":
        from ltx_trainer.uaclip.demand import amazon_optimal_attrs

        attrs = amazon_optimal_attrs()
        return {"demand_score": amazon_demand_score(attrs), "attributes": attrs}
    attrs = {"uniqueness": 0.55, "aesthetic": 0.58}
    return {"demand_score": airbnb_demand_score(attrs), "attributes": attrs}
