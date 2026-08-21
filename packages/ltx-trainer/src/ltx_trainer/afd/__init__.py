"""Adversarial Flow Distillation for autoregressive video (Luo et al., arXiv:2605.26105)."""

from ltx_trainer.afd.config import AFDConfig
from ltx_trainer.afd.diffusion_nft import (
    VelocityFieldStub,
    afd_student_loss,
    diffusion_nft_loss,
    forward_noisy_state,
    forward_velocity_target,
    nft_velocity_operators,
    prior_regularization,
)
from ltx_trainer.afd.discriminator import (
    PromptPairedDiscriminator,
    advantage_score,
    batch_baseline,
    bradley_terry_loss,
    density_ratio_logit,
    normalize_advantages_to_weights,
)
from ltx_trainer.afd.pipeline import (
    evaluation_demo,
    framework_card,
    table_bt_vs_gan_loss,
    table_discriminator_lr_ablation,
    table_physics_metrics,
    table_vbench_causal_forcing,
    table_vbench_self_forcing,
    training_step_demo,
)

__all__ = [
    "AFDConfig",
    "PromptPairedDiscriminator",
    "VelocityFieldStub",
    "advantage_score",
    "afd_student_loss",
    "batch_baseline",
    "bradley_terry_loss",
    "density_ratio_logit",
    "diffusion_nft_loss",
    "evaluation_demo",
    "forward_noisy_state",
    "forward_velocity_target",
    "framework_card",
    "nft_velocity_operators",
    "normalize_advantages_to_weights",
    "prior_regularization",
    "table_bt_vs_gan_loss",
    "table_discriminator_lr_ablation",
    "table_physics_metrics",
    "table_vbench_causal_forcing",
    "table_vbench_self_forcing",
    "training_step_demo",
]
