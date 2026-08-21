"""HoloFair: T2I fairness evaluation + Fair-GRPO debiasing (Chen et al., arXiv:2605.24687)."""

from ltx_trainer.holofair.config import HoloFairConfig
from ltx_trainer.holofair.metrics import (
    attribute_entropy,
    context_mean_diversity,
    context_robust_diversity,
    intrinsic_diversity,
    mgbi_score,
    normalized_entropy,
)
from ltx_trainer.holofair.pipeline import (
    demo_balanced_counts,
    evaluate_from_count_tables,
    fair_grpo_training_step,
)
from ltx_trainer.holofair.prompts import biased_gen_prompt, neutral_prompts, semantic_prompts
from ltx_trainer.holofair.rewards import (
    aggregate_image_reward,
    fair_grpo_rewards_for_batch,
    normalize_advantage,
)
from ltx_trainer.holofair.taxonomy import age_label, counts_from_labels, empty_counts

_LAZY_SPAFREQ = ("SpaFreqFusion", "SpaFreqHead", "frequency_view", "fuse_spatial_frequency")

__all__ = [
    "HoloFairConfig",
    "SpaFreqFusion",
    "SpaFreqHead",
    "age_label",
    "aggregate_image_reward",
    "attribute_entropy",
    "biased_gen_prompt",
    "context_mean_diversity",
    "context_robust_diversity",
    "counts_from_labels",
    "demo_balanced_counts",
    "empty_counts",
    "evaluate_from_count_tables",
    "fair_grpo_rewards_for_batch",
    "fair_grpo_training_step",
    "frequency_view",
    "fuse_spatial_frequency",
    "intrinsic_diversity",
    "mgbi_score",
    "neutral_prompts",
    "normalize_advantage",
    "normalized_entropy",
    "semantic_prompts",
    *_LAZY_SPAFREQ,
]


def __getattr__(name: str):
    if name in _LAZY_SPAFREQ:
        from ltx_trainer.holofair import spafreq

        return getattr(spafreq, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
