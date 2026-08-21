"""TT-SAC: test-time self-adaptive conditioning for talking heads (Zhang et al., arXiv:2605.25488)."""

from ltx_trainer.ttsac.config import TTSACConfig
from ltx_trainer.ttsac.operator import (
    apply_tt_sac,
    encode_generated_frames,
    generator_encoder_compose,
    monte_carlo_conditioning,
    refine_conditioning,
)
from ltx_trainer.ttsac.theory import (
    aggregated_covariance_diagonal,
    bias_variance_decomposition,
    bias_variance_prop3_scalar,
    empirical_cov_of_aggregate_mean,
    iid_aggregate_variance,
    identity_self_consistency_residual,
    lemma1_output_deviation_squared_bound,
    optimal_k_tradeoff,
)
from ltx_trainer.ttsac.pipeline import (
    evaluation_demo,
    framework_card,
    table_celebv_hq,
    table_hallo,
    table_k_sensitivity,
    table_ravdess_sonic,
    training_step_demo,
)

__all__ = [
    "TTSACConfig",
    "aggregated_covariance_diagonal",
    "apply_tt_sac",
    "bias_variance_decomposition",
    "bias_variance_prop3_scalar",
    "empirical_cov_of_aggregate_mean",
    "encode_generated_frames",
    "evaluation_demo",
    "framework_card",
    "generator_encoder_compose",
    "iid_aggregate_variance",
    "identity_self_consistency_residual",
    "lemma1_output_deviation_squared_bound",
    "monte_carlo_conditioning",
    "optimal_k_tradeoff",
    "refine_conditioning",
    "table_celebv_hq",
    "table_hallo",
    "table_k_sensitivity",
    "table_ravdess_sonic",
    "training_step_demo",
]
