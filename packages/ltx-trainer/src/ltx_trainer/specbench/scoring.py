"""Tiered IR scoring for SpecBench (paper §2.3.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ltx_trainer.specbench.config import SpecBenchConfig
from ltx_trainer.specbench.judging import MatchPair, match_predictions
from ltx_trainer.specbench.spi import SPITriple, decompose_deficiency
from ltx_trainer.specbench.tasks import GoldDeficiency, SpecBenchTask
from ltx_trainer.specbench.taxonomy import GoldTier


@dataclass
class TaskScore:
    task_id: str
    repository: str
    n_gold: int
    n_core: int
    n_extended: int
    prediction_budget: int
    n_predictions_used: int
    core_matches: int
    extended_matches: int
    weighted_score: float
    max_weighted_score: float
    accuracy: float
    matched_gold_ids: list[int] = field(default_factory=list)
    judge_jaccard: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "repository": self.repository,
            "n_gold": self.n_gold,
            "n_core": self.n_core,
            "n_extended": self.n_extended,
            "prediction_budget": self.prediction_budget,
            "n_predictions_used": self.n_predictions_used,
            "core_matches": self.core_matches,
            "extended_matches": self.extended_matches,
            "weighted_score": round(self.weighted_score, 4),
            "max_weighted_score": round(self.max_weighted_score, 4),
            "accuracy": round(self.accuracy, 4),
            "matched_gold_ids": self.matched_gold_ids,
            "judge_jaccard": round(self.judge_jaccard, 4),
        }


def prediction_budget(n_gold: int, cfg: SpecBenchConfig) -> int:
    return int(math.ceil(cfg.prediction_budget_multiplier * n_gold))


def tier_weight(tier: GoldTier, cfg: SpecBenchConfig) -> float:
    return cfg.core_weight if tier == GoldTier.CORE else cfg.extended_weight


def label_gold_tier(item: GoldDeficiency, cfg: SpecBenchConfig) -> GoldTier:
    """Recompute tier from expert scores when needed (stub uses pre-labeled tier)."""
    if not item.expert_scores:
        return item.tier
    mean = sum(item.expert_scores) / len(item.expert_scores)
    endorse = sum(1 for s in item.expert_scores if s >= cfg.core_likert_threshold)
    frac = endorse / len(item.expert_scores)
    if mean >= cfg.core_likert_threshold and frac >= cfg.core_endorsement_fraction:
        return GoldTier.CORE
    return GoldTier.EXTENDED


def score_task(
    task: SpecBenchTask,
    predictions: list[str],
    cfg: SpecBenchConfig,
    *,
    pred_spis: list[SPITriple] | None = None,
    forced_pairs: list[MatchPair] | None = None,
) -> TaskScore:
    """Score agent predictions against gold with bounded budget and tier weights."""
    gold = task.gold
    n_gold = len(gold)
    budget = prediction_budget(n_gold, cfg)
    trimmed = predictions[:budget]

    if pred_spis is None:
        pred_spis = [decompose_deficiency(p) for p in trimmed]
    else:
        pred_spis = pred_spis[:budget]

    gold_spi = [(g.id, g.spi) for g in gold]
    from ltx_trainer.specbench.judging import inter_trial_jaccard

    if forced_pairs is not None:
        pairs = forced_pairs
        jacc = 1.0
    else:
        pairs = match_predictions(pred_spis, gold_spi, cfg)
        jacc = inter_trial_jaccard(pred_spis, gold_spi, cfg)

    gold_by_id = {g.id: g for g in gold}
    matched_ids = [p.gold_id for p in pairs]
    core_matches = sum(1 for gid in matched_ids if gold_by_id[gid].tier == GoldTier.CORE)
    ext_matches = sum(1 for gid in matched_ids if gold_by_id[gid].tier == GoldTier.EXTENDED)

    weighted = 0.0
    max_weighted = 0.0
    for g in gold:
        w = tier_weight(g.tier, cfg)
        max_weighted += w
        if g.id in matched_ids:
            weighted += w

    accuracy = weighted / max_weighted if max_weighted > 0 else 0.0
    n_core = sum(1 for g in gold if g.tier == GoldTier.CORE)
    n_ext = n_gold - n_core

    return TaskScore(
        task_id=task.task_id,
        repository=task.repository,
        n_gold=n_gold,
        n_core=n_core,
        n_extended=n_ext,
        prediction_budget=budget,
        n_predictions_used=len(trimmed),
        core_matches=core_matches,
        extended_matches=ext_matches,
        weighted_score=weighted,
        max_weighted_score=max_weighted,
        accuracy=accuracy,
        matched_gold_ids=sorted(matched_ids),
        judge_jaccard=jacc,
    )


def aggregate_accuracy(scores: list[TaskScore]) -> float:
    total_w = sum(s.weighted_score for s in scores)
    total_max = sum(s.max_weighted_score for s in scores)
    return total_w / total_max if total_max > 0 else 0.0
