"""Occlusion-based interpretability (Section 4.2, Figure 2–3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.uaclip.demand import Platform, utility_regularized_similarity
from ltx_trainer.uaclip.similarity import clip_similarity
from ltx_trainer.uaclip.visual_attrs import VisualAttributes


@dataclass
class OcclusionResult:
    grid_h: int
    grid_w: int
    delta_map: np.ndarray
    mean_delta: float

    def to_dict(self) -> dict:
        return {
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "mean_delta": round(self.mean_delta, 6),
            "max_delta": round(float(self.delta_map.max()), 6),
        }


def patch_occlusion_sensitivity(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    attrs: VisualAttributes,
    platform: Platform,
    *,
    grid: tuple[int, int] = (4, 4),
    utility_aware: bool = True,
    eta: float = 0.5,
    mask_penalty: float = 0.15,
) -> OcclusionResult:
    """
    Patch occlusion on embedding space: masking reduces alignment score.

    Δ_r = Score(v,t) - Score(v^{-r}, t); warmer = larger drop.
    """
    gh, gw = grid
    base_attrs = attrs.to_dict(platform)
    base_sim = clip_similarity(image_emb, text_emb)
    if utility_aware:
        base_score = utility_regularized_similarity(base_sim, base_attrs, platform, eta=eta)
    else:
        base_score = base_sim

    deltas = np.zeros((gh, gw))
    for i in range(gh):
        for j in range(gw):
            perturbed = VisualAttributes(
                colorfulness=max(0.0, attrs.colorfulness - mask_penalty),
                brightness=max(0.0, attrs.brightness - mask_penalty),
                symmetry=max(0.0, attrs.symmetry - mask_penalty),
                aesthetic=max(0.0, attrs.aesthetic - mask_penalty),
                uniqueness=max(0.0, attrs.uniqueness - mask_penalty),
            )
            p_attrs = perturbed.to_dict(platform)
            p_emb = image_emb * (1.0 - mask_penalty)
            p_sim = clip_similarity(p_emb, text_emb)
            if utility_aware:
                p_score = utility_regularized_similarity(p_sim, p_attrs, platform, eta=eta)
            else:
                p_score = p_sim
            deltas[i, j] = base_score - p_score

    return OcclusionResult(
        grid_h=gh,
        grid_w=gw,
        delta_map=deltas,
        mean_delta=float(deltas.mean()),
    )
