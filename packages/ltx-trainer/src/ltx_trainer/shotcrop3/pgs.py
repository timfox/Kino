"""Pseudo-label generation strategy (PGS, Sec. 3.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.shotcrop3.config import PGSThresholds
from ltx_trainer.shotcrop3.geometry import iou


@dataclass(frozen=True)
class ProposalScores:
    mllm: float
    clip: float
    aesthetic: float

    @property
    def mean(self) -> float:
        return float((self.mllm + self.clip + self.aesthetic) / 3.0)


def score_proposal(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    shot_type: str,
    model_quality: float,
    rng: np.random.Generator,
) -> ProposalScores:
    """Synthetic PGS scores correlated with IoU and model stage."""
    align = iou(pred, gt)
    noise = rng.uniform(-0.05, 0.05)
    base = 0.35 + 0.55 * align + 0.1 * model_quality + noise
    clip_hint = 0.4 + 0.5 * align + (0.05 if shot_type == "medium" else 0.0)
    aes = 0.3 + 0.6 * align + 0.05 * model_quality
    return ProposalScores(
        mllm=float(np.clip(base, 0.0, 1.0)),
        clip=float(np.clip(clip_hint, 0.0, 1.0)),
        aesthetic=float(np.clip(aes, 0.0, 1.0)),
    )


def dominates(a: ProposalScores, b: ProposalScores) -> bool:
    return a.mllm > b.mllm and a.clip > b.clip and a.aesthetic > b.aesthetic


def select_pseudo_label(
    proposals: dict[str, np.ndarray],
    gts: np.ndarray | dict[str, np.ndarray],
    *,
    shot_type: str,
    model_qualities: dict[str, float],
    thresholds: PGSThresholds,
    rng: np.random.Generator,
) -> tuple[str | None, str]:
    """Return selected model name or None, plus status."""
    gt = gts if isinstance(gts, np.ndarray) else gts[shot_type]
    scored: dict[str, ProposalScores] = {}
    for name, box in proposals.items():
        scored[name] = score_proposal(
            box,
            gt,
            shot_type=shot_type,
            model_quality=model_qualities[name],
            rng=rng,
        )

    names = list(scored.keys())
    for candidate in names:
        others = [n for n in names if n != candidate]
        if all(dominates(scored[candidate], scored[o]) for o in others):
            s = scored[candidate]
            if s.mllm >= thresholds.tau_high and s.clip >= thresholds.tau_high and s.aesthetic >= thresholds.tau_high:
                return candidate, "pseudo"

    if all(s.mean < thresholds.tau_low for s in scored.values()):
        return None, "hard"

    return None, "reject"
