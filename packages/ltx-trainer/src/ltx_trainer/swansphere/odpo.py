"""Multi-objective Online DPO reward (Sec. 3.4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.swansphere.config import SwanSphereConfig


@dataclass
class ODPOScores:
    spatial: float
    semantic: float
    fidelity: float

    @property
    def total(self) -> float:
        cfg = SwanSphereConfig()
        return (
            cfg.odpo_lambda_spatial * self.spatial
            + cfg.odpo_lambda_semantic * self.semantic
            + cfg.odpo_lambda_fidelity * self.fidelity
        )


def normalize_score(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def odpo_reward(
    spatial_err: float,
    semantic_sim: float,
    aesthetic_dist: float,
    *,
    cfg: SwanSphereConfig | None = None,
) -> float:
    """
    Eq. 4 — lower spatial_err and aesthetic_dist are better; higher semantic_sim is better.
    Inputs are raw metrics; returns weighted score in [0,1] after normalization.
    """
    cfg = cfg or SwanSphereConfig()
    scores = ODPOScores(
        spatial=normalize_score(-spatial_err, -2.0, 0.0),
        semantic=normalize_score(semantic_sim, 0.0, 1.0),
        fidelity=normalize_score(-aesthetic_dist, -1.0, 0.0),
    )
    return scores.total


def rank_candidates(raw_metrics: list[tuple[float, float, float]]) -> tuple[int, int]:
    """Return (winner_idx, loser_idx) from list of (spatial_err, semantic, aesthetic)."""
    rewards = [odpo_reward(s, m, a) for s, m, a in raw_metrics]
    winner = max(range(len(rewards)), key=lambda i: rewards[i])
    loser = min(range(len(rewards)), key=lambda i: rewards[i])
    return winner, loser
