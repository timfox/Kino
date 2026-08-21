"""Ensemble LLM judging for SpecBench SPI matching (paper §2.3.3)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.specbench.config import SpecBenchConfig
from ltx_trainer.specbench.spi import SPITriple, spi_subject_predicate_match, token_overlap


@dataclass(frozen=True)
class MatchPair:
    pred_index: int
    gold_id: int


def _trial_thresholds(cfg: SpecBenchConfig) -> list[tuple[float, float]]:
    """Four trial-specific subject/predicate thresholds (stub ensemble)."""
    base_s, base_p = cfg.spi_subject_overlap, cfg.spi_predicate_overlap
    return [
        (base_s + 0.06, base_p + 0.05),
        (base_s, base_p),
        (base_s - 0.04, base_p - 0.03),
        (base_s - 0.02, base_p + 0.02),
    ]


def trial_matches(
    pred_spis: list[SPITriple],
    gold_spis: list[tuple[int, SPITriple]],
    cfg: SpecBenchConfig,
    *,
    trial_idx: int,
) -> list[MatchPair]:
    sub_t, pred_t = _trial_thresholds(cfg)[trial_idx % cfg.judge_trials]
    pairs: list[MatchPair] = []
    used_gold: set[int] = set()
    for pi, pred in enumerate(pred_spis):
        best_gold: int | None = None
        best_score = -1.0
        for gid, gspi in gold_spis:
            if gid in used_gold:
                continue
            if not spi_subject_predicate_match(pred, gspi, subject_threshold=sub_t, predicate_threshold=pred_t):
                continue
            score = token_overlap(pred.normalized_subject_tokens(), gspi.normalized_subject_tokens())
            score += token_overlap(pred.normalized_predicate_tokens(), gspi.normalized_predicate_tokens())
            if score > best_score:
                best_score = score
                best_gold = gid
        if best_gold is not None:
            pairs.append(MatchPair(pred_index=pi, gold_id=best_gold))
            used_gold.add(best_gold)
    return pairs


def ensemble_match(
    pred_spi: SPITriple,
    gold_spi: SPITriple,
    cfg: SpecBenchConfig,
) -> bool:
    """Majority vote across judge trials for a single pair."""
    votes = 0
    for sub_t, pred_t in _trial_thresholds(cfg):
        if spi_subject_predicate_match(
            pred_spi, gold_spi, subject_threshold=sub_t, predicate_threshold=pred_t
        ):
            votes += 1
    return votes >= cfg.judge_majority


def match_predictions(
    pred_spis: list[SPITriple],
    gold: list[tuple[int, SPITriple]],
    cfg: SpecBenchConfig,
) -> list[MatchPair]:
    """Greedy one-to-one matching with ensemble majority per candidate pair."""
    accepted: list[MatchPair] = []
    used_gold: set[int] = set()
    for pi, pred in enumerate(pred_spis):
        candidates: list[tuple[int, float]] = []
        for gid, gspi in gold:
            if gid in used_gold:
                continue
            if not ensemble_match(pred, gspi, cfg):
                continue
            score = token_overlap(pred.normalized_subject_tokens(), gspi.normalized_subject_tokens())
            score += token_overlap(pred.normalized_predicate_tokens(), gspi.normalized_predicate_tokens())
            candidates.append((gid, score))
        if not candidates:
            continue
        gid, _ = max(candidates, key=lambda x: x[1])
        accepted.append(MatchPair(pred_index=pi, gold_id=gid))
        used_gold.add(gid)
    return accepted


def jaccard_similarity(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def inter_trial_jaccard(
    pred_spis: list[SPITriple],
    gold: list[tuple[int, SPITriple]],
    cfg: SpecBenchConfig,
) -> float:
    """Median pairwise Jaccard across judge trials (Figure 6 proxy)."""
    trial_sets: list[set[tuple[int, int]]] = []
    for ti in range(cfg.judge_trials):
        pairs = trial_matches(pred_spis, gold, cfg, trial_idx=ti)
        trial_sets.append({(p.pred_index, p.gold_id) for p in pairs})
    sims: list[float] = []
    for i in range(len(trial_sets)):
        for j in range(i + 1, len(trial_sets)):
            sims.append(jaccard_similarity(trial_sets[i], trial_sets[j]))
    if not sims:
        return 1.0
    sims.sort()
    mid = len(sims) // 2
    return sims[mid] if len(sims) % 2 else (sims[mid - 1] + sims[mid]) / 2
