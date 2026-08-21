"""VLM checklist scorers and rescaling for WBENCH sub-metrics (Sec. 4, Appendix C)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from torch import Tensor


def rescale_0_100(x: float, *, src_min: float = 0.0, src_max: float = 1.0) -> float:
    """Linear map to [0, 100] as in Sec. 4."""
    if src_max <= src_min:
        return 0.0
    t = (x - src_min) / (src_max - src_min)
    return max(0.0, min(100.0, t * 100.0))


def hpsv3_norm(raw_mean: float, p1: float = 5.21, p99: float = 8.66) -> float:
    """Appendix C.2.6 — percentile-normalized HPSv3 reward."""
    if p99 <= p1:
        return 0.0
    t = (raw_mean - p1) / (p99 - p1)
    return rescale_0_100(max(0.0, min(1.0, t)))


def temporal_flickering_score(mae_per_pair: float) -> float:
    """Appendix C.2.3 — invert pixel MAE to [0, 100]."""
    return max(0.0, min(100.0, (255.0 - mae_per_pair) / 255.0 * 100.0))


def vlm_binary_checklist_score(
    checks: Sequence[bool],
    *,
    scale: float = 20.0,
) -> float:
    """Event editing / subject action: one point per matching binary check → ×20 for [0, 100]."""
    if not checks:
        return 0.0
    return float(sum(1 for c in checks if c) * scale)


def event_editing_turn_score(
    static_scene: bool,
    event_occurs: bool,
    event_complete: bool,
    detail_accurate: bool,
    no_anomaly: bool,
) -> float:
    """Five progressive checks (Appendix C.4.2); expected No, Yes, Yes, Yes, No."""
    checks = [
        not static_scene,
        event_occurs,
        event_complete,
        detail_accurate,
        no_anomaly,
    ]
    return vlm_binary_checklist_score(checks)


def subject_action_turn_score(
    subject_idle: bool,
    action_occurs: bool,
    action_complete: bool,
    detail_accurate: bool,
    natural_motion: bool,
) -> float:
    """Subject action template (Appendix C.4.3)."""
    checks = [
        not subject_idle,
        action_occurs,
        action_complete,
        detail_accurate,
        natural_motion,
    ]
    return vlm_binary_checklist_score(checks)


def perspective_switching_turn_score(
    transition_visible: bool,
    target_consistent: bool,
    quality_compliant: bool,
) -> float:
    """All three must hold (Appendix C.4.4); 0 or 100 per turn."""
    return 100.0 if (transition_visible and target_consistent and quality_compliant) else 0.0


def causal_fidelity_case_score(
    track1: float,
    track2_dims: Sequence[float] | None = None,
    *,
    src_max: float = 3.0,
) -> float:
    """Appendix C.6.1 — mean of Track 1 and mean Track 2, normalized to [0, 100]."""
    if track2_dims:
        t2 = sum(track2_dims) / len(track2_dims)
        raw = (track1 + t2) / 2.0
    else:
        raw = track1
    return rescale_0_100(raw, src_min=0.0, src_max=src_max)


def gated_spatial_consistency(
    return_similarity: float,
    min_intermediate_similarity: float,
    tau: float = 0.15,
) -> float:
    """Eq. (15) — motion gate suppresses static false positives."""
    gate = min(1.0, 1.0 - min_intermediate_similarity / tau)
    return rescale_0_100(return_similarity * gate)


def mean_sub_metrics(scores: dict[str, float]) -> float:
    """Dimension average over sub-metrics already on [0, 100]."""
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def batch_adjacent_cosine(features: "Tensor") -> float:
    """Smoke for subject/background consistency (DINO/CLIP adjacent-frame mean)."""
    import torch

    if features.shape[0] < 2:
        return 100.0
    feats = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    sims = (feats[1:] * feats[:-1]).sum(dim=-1)
    return rescale_0_100(float(sims.mean()), src_min=-1.0, src_max=1.0)
