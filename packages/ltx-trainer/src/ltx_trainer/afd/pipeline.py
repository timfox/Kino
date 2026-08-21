"""AFD framework card and reference experiment tables."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.afd.config import AFDConfig
from ltx_trainer.afd.diffusion_nft import (
    VelocityFieldStub,
    afd_student_loss,
    forward_noisy_state,
    forward_velocity_target,
)
from ltx_trainer.afd.discriminator import (
    PromptPairedDiscriminator,
    advantage_score,
    bradley_terry_loss,
    density_ratio_logit,
    normalize_advantages_to_weights,
)


def framework_card(cfg: AFDConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AFDConfig()
    return {
        "name": "AFD",
        "paper": "arXiv:2605.26105",
        "title": "On-Policy Adversarial Flow Distillation for Autoregressive Video Generation",
        "teacher": f"{cfg.teacher_api} (sampling-only API)",
        "students": list(cfg.student_backbones),
        "stages": [
            "on-policy rollouts + teacher videos",
            "Bradley–Terry discriminator",
            "DiffusionNFT forward-process update",
        ],
        "requires": "clean teacher videos and student rollouts only (no scores/latents)",
        "hyperparams": {
            "nft_beta": cfg.nft_beta,
            "prior_weight": cfg.prior_weight,
            "discriminator_lr": cfg.discriminator_lr,
            "student_lr": cfg.learning_rate,
        },
    }


def table_vbench_self_forcing() -> dict[str, dict[str, float]]:
    """Table 1 — Self-Forcing VBench after continual adaptation."""
    return {
        "base": {"physics_total": 68.49, "general_total": 36.51, "dynamic_degree": 91.67},
        "sft": {"physics_total": 58.69, "general_total": 37.62, "dynamic_degree": 44.44},
        "gan": {"physics_total": 83.83, "general_total": 60.95, "dynamic_degree": 61.11},
        "dmd": {"physics_total": 81.88, "general_total": 59.98, "dynamic_degree": 52.78},
        "afd": {"physics_total": 87.55, "general_total": 60.83, "dynamic_degree": 79.17},
    }


def table_vbench_causal_forcing() -> dict[str, dict[str, float]]:
    """Table 1 — Causal-Forcing VBench."""
    return {
        "base": {"physics_total": 86.76, "general_total": 59.03, "dynamic_degree": 87.50},
        "sft": {"physics_total": 76.24, "general_total": 59.40, "dynamic_degree": 29.17},
        "gan": {"physics_total": 83.07, "general_total": 59.44, "dynamic_degree": 66.67},
        "dmd": {"physics_total": 82.52, "general_total": 60.32, "dynamic_degree": 58.33},
        "afd": {"physics_total": 88.52, "general_total": 59.83, "dynamic_degree": 88.89},
    }


def table_physics_metrics() -> dict[str, dict[str, dict[str, float]]]:
    """Table 2 — VideoAlign-MQ and VideoPhy-2-PC."""
    return {
        "self_forcing": {
            "base": {"motion_quality": 0.341, "physical_consistency": 4.04},
            "sft": {"motion_quality": 0.296, "physical_consistency": 3.72},
            "gan": {"motion_quality": 0.541, "physical_consistency": 4.10},
            "dmd": {"motion_quality": 0.420, "physical_consistency": 3.72},
            "afd": {"motion_quality": 0.605, "physical_consistency": 4.20},
        },
        "causal_forcing": {
            "base": {"motion_quality": 0.520, "physical_consistency": 4.16},
            "sft": {"motion_quality": 0.499, "physical_consistency": 4.17},
            "gan": {"motion_quality": 0.582, "physical_consistency": 4.16},
            "dmd": {"motion_quality": 0.661, "physical_consistency": 4.14},
            "afd": {"motion_quality": 0.661, "physical_consistency": 4.24},
        },
    }


def table_discriminator_lr_ablation() -> dict[str, str]:
    """Fig. 5 — discriminator learning-rate regimes."""
    return {
        "0": "reward hacking (stale discriminator)",
        "1e-6": "saturation — r_φ → 1",
        "5e-6": "informative adaptive feedback",
        "1e-5": "informative adaptive feedback",
        "5e-5": "suppressed learning — r_φ → 0",
    }


def table_bt_vs_gan_loss() -> dict[str, dict[str, float]]:
    """Fig. 6 — BT vs GAN discriminator on Causal-Forcing motion dims."""
    return {
        "bt_loss": {
            "motion_smoothness": 98.89,
            "dynamic_degree": 88.89,
            "temporal_flickering": 98.41,
            "human_action": 80.00,
            "spatial_relation": 77.39,
        },
        "gan_loss": {
            "motion_smoothness": 98.53,
            "dynamic_degree": 79.17,
            "temporal_flickering": 99.07,
            "human_action": 80.00,
            "spatial_relation": 79.53,
        },
    }


def training_step_demo(
    cfg: AFDConfig | None = None,
    *,
    batch_size: int = 4,
    device: str = "cpu",
) -> dict[str, float]:
    """One AFD iteration smoke (Algorithm 1)."""
    cfg = cfg or AFDConfig()
    dev = torch.device(device)
    b = batch_size
    feat_dim = 64

    teacher_feat = torch.randn(b, feat_dim, device=dev)
    student_feat = torch.randn(b, feat_dim, device=dev)
    prompt_feat = torch.randn(b, feat_dim, device=dev)

    disc = PromptPairedDiscriminator(cfg).to(dev)
    d_t = disc(teacher_feat, prompt_feat)
    d_s = disc(student_feat, prompt_feat)
    loss_d = float(bradley_terry_loss(d_t, d_s).item())

    adv = advantage_score(d_s)
    weights = normalize_advantages_to_weights(adv, clip_max=cfg.advantage_clip_max)

    x0 = torch.randn(b, 32, device=dev)
    t = torch.rand(b, device=dev)
    noise = torch.randn_like(x0)
    x_t = forward_noisy_state(x0, t, noise)
    v_target = forward_velocity_target(x0, t, noise)

    student = VelocityFieldStub(dim=32).to(dev)
    ref = VelocityFieldStub(dim=32).to(dev)
    ref.load_state_dict(student.state_dict())
    v_theta = student(x_t, t)
    v_ref = ref(x_t, t)
    loss_s = float(afd_student_loss(v_theta, v_target, v_ref, weights, cfg).item())

    return {
        "discriminator_loss": loss_d,
        "student_loss": loss_s,
        "mean_weight": float(weights.mean()),
        "mean_advantage": float(adv.mean()),
        "density_ratio_logit": float(density_ratio_logit(torch.sigmoid(d_s).detach()).mean()),
    }


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    """Smoke + reference table anchors."""
    step = training_step_demo(device=device)
    sf = table_vbench_self_forcing()
    return {
        **step,
        "afd_physics_self_forcing": sf["afd"]["physics_total"],
        "afd_dynamic_degree_causal": table_vbench_causal_forcing()["afd"]["dynamic_degree"],
        "afd_mq_self_forcing": table_physics_metrics()["self_forcing"]["afd"]["motion_quality"],
        "bt_dynamic_gain": (
            table_bt_vs_gan_loss()["bt_loss"]["dynamic_degree"]
            - table_bt_vs_gan_loss()["gan_loss"]["dynamic_degree"]
        ),
    }
