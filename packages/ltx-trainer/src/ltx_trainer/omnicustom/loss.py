"""Flow matching + contrastive identity/timbre objectives (paper Eq. 8–10)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.omnicustom.config import OmniCustomConfig


@dataclass
class OmniCustomLossBreakdown:
    video_fm: float
    audio_fm: float
    identity_cl: float
    timbre_cl: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "video_fm": round(self.video_fm, 6),
            "audio_fm": round(self.audio_fm, 6),
            "identity_cl": round(self.identity_cl, 6),
            "timbre_cl": round(self.timbre_cl, 6),
            "total": round(self.total, 6),
        }


def flow_matching_loss(v_pred: np.ndarray, v_target: np.ndarray) -> float:
    """L2 flow matching ‖vθ − (z1 − z0)‖² averaged."""
    diff = v_pred.astype(np.float64) - v_target.astype(np.float64)
    return float(np.mean(diff * diff))


def contrastive_cl_loss(v_with_ref: np.ndarray, v_no_ref: np.ndarray) -> float:
    """Maximize dissimilarity: −‖v_ref − stopgrad(v_no_ref)‖² (paper Eq. 9)."""
    diff = v_with_ref.astype(np.float64) - v_no_ref.astype(np.float64)
    return float(-np.mean(diff * diff))


def total_omnicustom_loss(
    *,
    v_pred: np.ndarray,
    v_target: np.ndarray,
    a_pred: np.ndarray,
    a_target: np.ndarray,
    v_pred_no_ref: np.ndarray,
    a_pred_no_ref: np.ndarray,
    cfg: OmniCustomConfig | None = None,
) -> OmniCustomLossBreakdown:
    c = cfg or OmniCustomConfig()
    lv = flow_matching_loss(v_pred, v_target)
    la = flow_matching_loss(a_pred, a_target)
    li = contrastive_cl_loss(v_pred, v_pred_no_ref)
    lt = contrastive_cl_loss(a_pred, a_pred_no_ref)
    total = (
        c.lambda_video_fm * lv
        + c.lambda_audio_fm * la
        + c.lambda_identity_cl * li
        + c.lambda_timbre_cl * lt
    )
    return OmniCustomLossBreakdown(video_fm=lv, audio_fm=la, identity_cl=li, timbre_cl=lt, total=total)
