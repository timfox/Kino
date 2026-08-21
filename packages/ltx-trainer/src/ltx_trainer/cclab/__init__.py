"""CCLab: adversarial testing of congestion controllers (Chen et al., arXiv:2605.21915)."""

from ltx_trainer.cclab.config import (
    CCLabConfig,
    ENV_LEVEL_CC,
    FEATURE_LEVEL_CC,
    LEARNING_CC,
    NON_LEARNING_CC,
)
from ltx_trainer.cclab.metrics import bandwidth_utilization, cwnd_smoothness, max_utilization_degradation
from ltx_trainer.cclab.perturbations import (
    append_bandwidth_step,
    average_absolute_slope,
    log_scaled_cwnd_smoothness,
    perturb_min_rtt,
    perturbation_bounds,
)
from ltx_trainer.cclab.pipeline import (
    adversarial_step_demo,
    environment_degradation_vs_random,
    feature_degradation_summary,
    framework_card,
    table_adversarial_training,
    table_cwnd_smoothness,
    table_environment_bandwidth,
    table_feature_50pct,
    table_feature_5pct,
)
from ltx_trainer.cclab.rewards import (
    cc_reward,
    delay_factor,
    delay_penalty,
    improved_adversarial_reward,
    naive_adversarial_reward,
    queuing_delay,
    utilization_degradation_pct,
)

__all__ = [
    "CCLabConfig",
    "ENV_LEVEL_CC",
    "FEATURE_LEVEL_CC",
    "LEARNING_CC",
    "NON_LEARNING_CC",
    "adversarial_step_demo",
    "append_bandwidth_step",
    "average_absolute_slope",
    "bandwidth_utilization",
    "cc_reward",
    "cwnd_smoothness",
    "delay_factor",
    "delay_penalty",
    "environment_degradation_vs_random",
    "feature_degradation_summary",
    "framework_card",
    "improved_adversarial_reward",
    "log_scaled_cwnd_smoothness",
    "max_utilization_degradation",
    "naive_adversarial_reward",
    "perturb_min_rtt",
    "perturbation_bounds",
    "queuing_delay",
    "table_adversarial_training",
    "table_cwnd_smoothness",
    "table_environment_bandwidth",
    "table_feature_50pct",
    "table_feature_5pct",
    "utilization_degradation_pct",
]
