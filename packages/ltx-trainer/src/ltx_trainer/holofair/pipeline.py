"""Evaluation and Fair-GRPO glue (Fig. 2–3)."""

from __future__ import annotations

from ltx_trainer.holofair.config import HoloFairConfig
from ltx_trainer.holofair.metrics import mgbi_score
from ltx_trainer.holofair.rewards import fair_grpo_rewards_for_batch, normalize_advantage
from ltx_trainer.holofair.taxonomy import empty_counts


def evaluate_from_count_tables(
    neutral_counts: dict[str, dict[str, int]],
    semantic_counts: dict[str, dict[str, dict[str, int]]],
    *,
    cfg: HoloFairConfig | None = None,
) -> dict[str, float]:
    """End-to-end MGBI report from classifier count tables."""
    return mgbi_score(neutral_counts, semantic_counts, cfg=cfg)


def demo_balanced_counts(*, cfg: HoloFairConfig | None = None) -> dict[str, dict[str, int]]:
    """Uniformly balanced neutral counts for smoke tests."""
    cfg = cfg or HoloFairConfig()
    counts = empty_counts(cfg)
    for attr in cfg.attributes:
        cats = cfg.categories_for(attr)
        for c in cats:
            counts[attr][c] = 10
    return counts


def fair_grpo_training_step(
    label_batch: list[dict[str, str]],
    *,
    cfg: HoloFairConfig | None = None,
    running_mean: float | None = None,
    running_std: float | None = None,
) -> dict[str, object]:
    """Return per-image rewards and normalized advantages for one prompt group."""
    cfg = cfg or HoloFairConfig()
    rewards, counts = fair_grpo_rewards_for_batch(label_batch, cfg=cfg)
    advantages = normalize_advantage(rewards, mean=running_mean, std=running_std)
    return {
        "rewards": rewards,
        "advantages": advantages,
        "batch_counts": counts,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
    }
