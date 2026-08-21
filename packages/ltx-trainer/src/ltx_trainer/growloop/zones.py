"""Consensus–divergence aware evaluation (GrowLoop §3.2)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.growloop.config import EvaluationZone, GrowLoopConfig


def partition_zones(
    human_scores: list[list[int]],
    *,
    unanimous_only: bool = True,
) -> tuple[list[int], list[int]]:
    """Return (consensus_indices, divergence_indices) for per-case score lists."""
    consensus: list[int] = []
    divergence: list[int] = []
    for i, scores in enumerate(human_scores):
        if not scores:
            continue
        if unanimous_only:
            if len(set(scores)) == 1:
                consensus.append(i)
            else:
                divergence.append(i)
        elif max(scores) == min(scores):
            consensus.append(i)
        else:
            divergence.append(i)
    return consensus, divergence


def zone_aware_agreement(
    ai_scores: list[int],
    human_scores: list[list[int]],
    *,
    consensus_indices: list[int] | None = None,
) -> dict[str, float]:
    """Agreement rate on consensus zone; plausibility proxy on divergence zone."""
    if consensus_indices is None:
        consensus_indices, divergence_indices = partition_zones(human_scores)
    else:
        _, divergence_indices = partition_zones(human_scores)

    def _match(i: int) -> bool:
        hs = human_scores[i]
        if not hs:
            return False
        return ai_scores[i] == hs[0]

    def _plausible(i: int) -> bool:
        hs = human_scores[i]
        if not hs:
            return False
        lo, hi = min(hs), max(hs)
        return lo <= ai_scores[i] <= hi

    cons_n = len(consensus_indices)
    div_n = len(divergence_indices)
    cons_rate = sum(_match(i) for i in consensus_indices) / cons_n if cons_n else 0.0
    div_rate = sum(_plausible(i) for i in divergence_indices) / div_n if div_n else 0.0
    return {
        "consensus_agreement": cons_rate,
        "divergence_plausibility": div_rate,
        "consensus_n": float(cons_n),
        "divergence_n": float(div_n),
    }


def zones_smoke(cfg: GrowLoopConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GrowLoopConfig()
    # Toy: 3 unanimous, 2 divergent (Table 1 style)
    human = [[2], [0], [1], [0, 1, 2], [1, 2]]
    ai = [2, 0, 1, 2, 1]
    out = zone_aware_agreement(ai, human)
    out["inter_annotator_agreement_anchor"] = cfg.inter_annotator_agreement
    return out
