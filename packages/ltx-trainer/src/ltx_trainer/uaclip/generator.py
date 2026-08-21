"""Utility-Aware Generator: candidate ranking via U-CLIP (Section 5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.uaclip.demand import Platform, airbnb_demand_score, amazon_demand_score
from ltx_trainer.uaclip.similarity import clip_similarity, utility_aware_score
from ltx_trainer.uaclip.visual_attrs import VisualAttributes


@dataclass
class CandidateImage:
    candidate_id: str
    image_emb: np.ndarray
    attrs: VisualAttributes
    clip_sim: float = 0.0
    uclip_score: float = 0.0
    demand_score: float = 0.0
    fidelity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "clip_sim": round(self.clip_sim, 4),
            "uclip_score": round(self.uclip_score, 4),
            "demand_score": round(self.demand_score, 4),
            "fidelity": round(self.fidelity, 4),
        }


def score_candidates(
    candidates: list[CandidateImage],
    text_emb: np.ndarray,
    reference_emb: np.ndarray,
    platform: Platform,
    *,
    eta: float = 0.5,
    alpha_visual: float = 1.0,
    beta_semantic: float = 1.0,
) -> list[CandidateImage]:
    """Rank candidates by Utility-Aware CLIP score (Eq. 3)."""
    scored: list[CandidateImage] = []
    for c in candidates:
        sim = clip_similarity(c.image_emb, text_emb)
        attrs = c.attrs.to_dict(platform)
        if platform == "amazon":
            from ltx_trainer.uaclip.demand import amazon_log_sales_rank

            demand = amazon_demand_score(attrs)
            util = eta * amazon_log_sales_rank(attrs)
        else:
            from ltx_trainer.uaclip.demand import airbnb_log_occupancy

            demand = airbnb_demand_score(attrs)
            util = eta * airbnb_log_occupancy(attrs)
        uclip = utility_aware_score(sim, util, alpha_visual=alpha_visual, beta_semantic=beta_semantic)
        fid = clip_similarity(c.image_emb, reference_emb)
        scored.append(
            CandidateImage(
                candidate_id=c.candidate_id,
                image_emb=c.image_emb,
                attrs=c.attrs,
                clip_sim=float(sim),
                uclip_score=float(uclip),
                demand_score=float(demand),
                fidelity=float(fid),
            )
        )
    return sorted(scored, key=lambda x: x.uclip_score, reverse=True)


def select_best(
    candidates: list[CandidateImage],
    text_emb: np.ndarray,
    reference_emb: np.ndarray,
    platform: Platform,
    **kwargs: Any,
) -> CandidateImage:
    ranked = score_candidates(candidates, text_emb, reference_emb, platform, **kwargs)
    return ranked[0]
