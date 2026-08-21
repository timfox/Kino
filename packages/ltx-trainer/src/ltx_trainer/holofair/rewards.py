"""Fair-GRPO multi-attribute per-prompt reward (Sec. 3.4, Eq. 9–14)."""

from __future__ import annotations

import math
from collections import defaultdict

from ltx_trainer.holofair.config import HoloFairConfig


def base_fairness_reward(n_k: int, n_total: int, *, epsilon: float = 1e-6) -> float:
    """Eq. (9): log((N - N_k + eps) / (N_k + eps))."""
    return math.log((n_total - n_k + epsilon) / (n_k + epsilon))


def zero_centered_reward(
    category: str,
    counts: dict[str, int],
    *,
    cfg: HoloFairConfig | None = None,
    attribute: str = "gender",
) -> float:
    """Eq. (10)–(11): r_fair = r_base - mean(r_base)."""
    cfg = cfg or HoloFairConfig()
    cats = cfg.categories_for(attribute)
    n_total = sum(counts.get(c, 0) for c in cats)
    if n_total <= 0:
        return 0.0
    bases = {c: base_fairness_reward(counts.get(c, 0), n_total, epsilon=cfg.edit_reward_epsilon) for c in cats}
    mean_base = sum(bases.values()) / len(cats)
    return bases[category] - mean_base


def clipped_category_reward(
    category: str,
    counts: dict[str, int],
    *,
    cfg: HoloFairConfig | None = None,
    attribute: str = "gender",
) -> float:
    """Eq. (12): clip(r_fair, Rmin, Rmax)."""
    cfg = cfg or HoloFairConfig()
    r = zero_centered_reward(category, counts, cfg=cfg, attribute=attribute)
    return max(cfg.reward_min, min(cfg.reward_max, r))


def aggregate_image_reward(
    labels: dict[str, str],
    batch_counts: dict[str, dict[str, int]],
    *,
    cfg: HoloFairConfig | None = None,
) -> float:
    """Eq. (13): R(Ip) = sum_a w_a * r_clip(F(Ip), a)."""
    cfg = cfg or HoloFairConfig()
    total = 0.0
    for attr in cfg.attributes:
        cat = labels.get(attr)
        if cat is None or cat not in batch_counts[attr]:
            continue
        w = cfg.attribute_reward_weights.get(attr, 1.0)
        total += w * clipped_category_reward(cat, batch_counts[attr], cfg=cfg, attribute=attr)
    return total


def accumulate_batch_counts(
    label_batch: list[dict[str, str]],
    *,
    cfg: HoloFairConfig | None = None,
) -> dict[str, dict[str, int]]:
    """Count N^a_k within a prompt group of size N."""
    cfg = cfg or HoloFairConfig()
    counts: dict[str, dict[str, int]] = {a: {c: 0 for c in cfg.categories_for(a)} for a in cfg.attributes}
    for labels in label_batch:
        for attr in cfg.attributes:
            cat = labels.get(attr)
            if cat in counts[attr]:
                counts[attr][cat] += 1
    return counts


def normalize_advantage(
    rewards: list[float],
    *,
    mean: float | None = None,
    std: float | None = None,
    epsilon: float = 1e-6,
) -> list[float]:
    """Eq. (14): per-prompt group advantage normalization."""
    if not rewards:
        return []
    mu = mean if mean is not None else sum(rewards) / len(rewards)
    if std is None:
        var = sum((r - mu) ** 2 for r in rewards) / max(len(rewards), 1)
        sigma = math.sqrt(var)
    else:
        sigma = std
    return [(r - mu) / (sigma + epsilon) for r in rewards]


def fair_grpo_rewards_for_batch(
    label_batch: list[dict[str, str]],
    *,
    cfg: HoloFairConfig | None = None,
) -> tuple[list[float], dict[str, dict[str, int]]]:
    """Compute scalar R(Ip) for each image in a prompt group."""
    cfg = cfg or HoloFairConfig()
    counts = accumulate_batch_counts(label_batch, cfg=cfg)
    rewards = [aggregate_image_reward(lbl, counts, cfg=cfg) for lbl in label_batch]
    return rewards, counts
