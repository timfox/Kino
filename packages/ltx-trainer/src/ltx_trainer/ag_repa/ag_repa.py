"""Attribution-Guided REPA training objective (paper Eq. 10–12)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.ag_repa.bitc import bitc_score, dual_bitc_loss
from ltx_trainer.ag_repa.config import AgRepaConfig
from ltx_trainer.ag_repa.fog_a import FogAScore, attribution_weights, top_k_by_fog_a


@dataclass
class AgRepaLossBreakdown:
    flow_matching: float
    bitc: float
    repa_alignment: float
    total: float
    selected_layers: tuple[int, ...]
    layer_weights: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "flow_matching": round(self.flow_matching, 6),
            "bitc": round(self.bitc, 6),
            "repa_alignment": round(self.repa_alignment, 6),
            "total": round(self.total, 6),
            "selected_layers": list(self.selected_layers),
            "layer_weights": {str(k): round(v, 4) for k, v in self.layer_weights.items()},
        }


def flow_matching_loss(v_pred: np.ndarray, v_target: np.ndarray) -> float:
    diff = v_pred.astype(np.float64) - v_target.astype(np.float64)
    return float(np.mean(diff * diff))


def repa_layer_loss(projected: np.ndarray, teacher: np.ndarray) -> float:
    """1 − cos(hϕk(¯hk), T(x)) per layer (Eq. 12)."""
    return 1.0 - bitc_score(projected, teacher)


def select_ag_repa_layers(fog_scores: list[FogAScore], cfg: AgRepaConfig | None = None) -> list[int]:
    c = cfg or AgRepaConfig()
    return top_k_by_fog_a(fog_scores, c.top_k_layers)


def total_ag_repa_loss(
    *,
    v_pred: np.ndarray,
    v_target: np.ndarray,
    speech_proj: np.ndarray,
    speech_teacher: np.ndarray,
    audio_proj: np.ndarray,
    audio_teacher: np.ndarray,
    layer_projections: dict[int, np.ndarray],
    layer_teachers: dict[int, np.ndarray],
    fog_scores: list[FogAScore],
    cfg: AgRepaConfig | None = None,
) -> AgRepaLossBreakdown:
    c = cfg or AgRepaConfig()
    lfm = flow_matching_loss(v_pred, v_target)
    lbit = dual_bitc_loss(speech_proj, speech_teacher, audio_proj, audio_teacher)

    selected = select_ag_repa_layers(fog_scores, c)
    weights = attribution_weights(fog_scores, selected)
    repa = 0.0
    for layer in selected:
        w = weights[layer]
        repa += w * repa_layer_loss(layer_projections[layer], layer_teachers[layer])

    total = lfm + c.lambda_bit * lbit + c.lambda_repa_base * repa
    return AgRepaLossBreakdown(
        flow_matching=lfm,
        bitc=lbit,
        repa_alignment=repa,
        total=total,
        selected_layers=tuple(selected),
        layer_weights=weights,
    )
