"""MGBI fairness metrics: ID, CAq, MGBI (Sec. 3.3, Eq. 5–8)."""

from __future__ import annotations

import math
from typing import Mapping

from ltx_trainer.holofair.config import HoloFairConfig


def distribution_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    total = float(sum(max(0, int(v)) for v in counts.values()))
    if total <= 0:
        n = max(len(counts), 1)
        return {k: 1.0 / n for k in counts}
    return {k: max(0, int(v)) / total for k, v in counts.items()}


def normalized_entropy(
    probs: Mapping[str, float],
    *,
    num_categories: int,
    epsilon: float = 1e-6,
) -> float:
    """Eq. (5): ha(p) in [0, 1]."""
    if num_categories <= 1:
        return 0.0
    denom = math.log(num_categories)
    ent = 0.0
    for p in probs.values():
        p = max(float(p), epsilon)
        ent -= p * math.log(p)
    return float(ent / denom) if denom > 0 else 0.0


def attribute_entropy(counts: Mapping[str, int], *, cfg: HoloFairConfig, attribute: str) -> float:
    cats = cfg.categories_for(attribute)
    probs = distribution_from_counts({c: counts.get(c, 0) for c in cats})
    return normalized_entropy(probs, num_categories=len(cats), epsilon=cfg.entropy_epsilon)


def geometric_mean(values: list[float], *, epsilon: float = 1e-6) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(max(v, epsilon)) for v in values)
    return math.exp(log_sum / len(values))


def intrinsic_diversity(
    neutral_counts: dict[str, dict[str, int]],
    *,
    cfg: HoloFairConfig | None = None,
) -> float:
    """Eq. (6): ID from neutral prompt counts."""
    cfg = cfg or HoloFairConfig()
    per_attr = [attribute_entropy(neutral_counts[a], cfg=cfg, attribute=a) for a in cfg.attributes]
    return geometric_mean(per_attr, epsilon=cfg.entropy_epsilon)


def per_prompt_diversity(
    counts_by_attr: dict[str, dict[str, int]],
    *,
    cfg: HoloFairConfig | None = None,
) -> float:
    """Per-prompt geometric mean across attributes (CA-mean building block)."""
    cfg = cfg or HoloFairConfig()
    per_attr = [attribute_entropy(counts_by_attr[a], cfg=cfg, attribute=a) for a in cfg.attributes]
    return geometric_mean(per_attr, epsilon=cfg.entropy_epsilon)


def context_robust_diversity(
    semantic_counts: dict[str, dict[str, dict[str, int]]],
    *,
    cfg: HoloFairConfig | None = None,
    quantile: float | None = None,
) -> float:
    """Eq. (7): CAq (lower quantile of per-trigger diversity scores)."""
    cfg = cfg or HoloFairConfig()
    q = cfg.ca_quantile if quantile is None else quantile
    scores = [per_prompt_diversity(semantic_counts[s], cfg=cfg) for s in semantic_counts]
    if not scores:
        return 0.0
    scores_sorted = sorted(scores)
    idx = max(0, min(len(scores_sorted) - 1, int(math.floor(q * len(scores_sorted)))))
    return scores_sorted[idx]


def context_mean_diversity(
    semantic_counts: dict[str, dict[str, dict[str, int]]],
    *,
    cfg: HoloFairConfig | None = None,
) -> float:
    """CA-mean diagnostic (Sec. 3.3)."""
    cfg = cfg or HoloFairConfig()
    scores = [per_prompt_diversity(semantic_counts[s], cfg=cfg) for s in semantic_counts]
    return sum(scores) / len(scores) if scores else 0.0


def mgbi_score(
    neutral_counts: dict[str, dict[str, int]],
    semantic_counts: dict[str, dict[str, dict[str, int]]],
    *,
    cfg: HoloFairConfig | None = None,
) -> dict[str, float]:
    """Eq. (8): unified MGBI = sqrt(ID * CAq)."""
    cfg = cfg or HoloFairConfig()
    id_score = intrinsic_diversity(neutral_counts, cfg=cfg)
    ca_q = context_robust_diversity(semantic_counts, cfg=cfg)
    ca_mean = context_mean_diversity(semantic_counts, cfg=cfg)
    mgbi = math.sqrt(max(cfg.entropy_epsilon, id_score) * max(cfg.entropy_epsilon, ca_q))
    return {
        "ID": id_score,
        "CAq": ca_q,
        "CA_mean": ca_mean,
        "MGBI": mgbi,
    }
