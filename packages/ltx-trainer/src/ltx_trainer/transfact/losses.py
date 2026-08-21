"""TransFACT multi-task loss stubs (frame, stage, transferability)."""

from __future__ import annotations

import math
from typing import Sequence


def cross_entropy_log_prob(log_prob: float) -> float:
    return -math.log(max(log_prob, 1e-12))


def frame_classification_loss(log_probs: Sequence[float]) -> float:
    return sum(cross_entropy_log_prob(p) for p in log_probs) / max(len(log_probs), 1)


def transferability_loss(log_prob_positive: float) -> float:
    return cross_entropy_log_prob(log_prob_positive)


def total_transfact_loss(
    *,
    l_trans: float,
    l_frame: float,
    l_stage: float,
    l_cross: float = 0.0,
    l_smooth: float = 0.0,
    weight_smooth: float = 5.0,
) -> float:
    """L = Ltrans + Lframe + Lstage + Lcross + 5*Lsmooth (paper weights)."""
    return l_trans + l_frame + l_stage + l_cross + weight_smooth * l_smooth
