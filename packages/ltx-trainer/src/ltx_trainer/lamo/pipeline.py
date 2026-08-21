"""Training and evaluation glue for LaMo."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.lamo.config import LaMoConfig
from ltx_trainer.lamo.guidance import guidance_loss
from ltx_trainer.lamo.heatmaps import motion_drift_heatmap, motion_field_heatmap
from ltx_trainer.lamo.latent_motion import empirical_macro_drift, latent_delta
from ltx_trainer.lamo.losses import motion_drift_loss, predictor_loss, training_objective
from ltx_trainer.lamo.predictor import MotionFieldPredictor


def lamo_training_step(
    l_denoise: Tensor,
    x0_pred: Tensor,
    x0_target: Tensor,
    *,
    alpha_bar: Tensor,
    cfg: LaMoConfig | None = None,
) -> dict[str, Tensor | float]:
    """One backbone fine-tune step with Motion Drift Loss (Eq. 6)."""
    cfg = cfg or LaMoConfig()
    l_drift = motion_drift_loss(
        x0_pred,
        x0_target,
        tau=cfg.tau,
        epsilon=cfg.drift_epsilon,
    )
    loss = training_objective(l_denoise, l_drift, lambda_drift=cfg.lambda_drift, alpha_bar=alpha_bar)
    return {
        "loss": loss,
        "loss_denoise": l_denoise.detach(),
        "loss_drift": l_drift.detach(),
    }


def train_predictor_step(
    z: Tensor,
    cond: Tensor,
    predictor: MotionFieldPredictor,
    *,
    cfg: LaMoConfig | None = None,
    augment_noise: bool = False,
    sigma: float = 0.1,
) -> dict[str, Tensor]:
    """Train f_φ on clean latent pairs (Eq. 7)."""
    cfg = cfg or LaMoConfig()
    if z.dim() != 4:
        raise ValueError("z must be (T, C, H, W)")
    if torch.rand(1).item() < cfg.classifier_free_drop_prob:
        cond = predictor.null_cond.unsqueeze(0).expand(1, -1)
    losses: list[Tensor] = []
    fields: list[Tensor] = []
    for i in range(z.size(0) - cfg.tau):
        zi = z[i]
        if augment_noise and torch.rand(1).item() < cfg.predictor_aug_prob:
            zi = zi + sigma * torch.randn_like(zi)
        target = z[i + cfg.tau] - z[i]
        field = predictor(zi.unsqueeze(0), cond)
        losses.append(
            predictor_loss(field, target.unsqueeze(0), cosine_weight=cfg.predictor_cosine_weight)
        )
        fields.append(field.detach())
    loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=z.device)
    motion_field = fields[0] if fields else torch.zeros_like(z[:1])
    return {"loss_phi": loss, "motion_field": motion_field}


def evaluate_motion_readouts(
    z: Tensor,
    predictor: MotionFieldPredictor,
    cond: Tensor,
    *,
    cfg: LaMoConfig | None = None,
) -> dict[str, Any]:
    """Demo: macro drift norm + heatmap peaks for interpretability."""
    cfg = cfg or LaMoConfig()
    if z.dim() == 5:
        z = z[0]
    drift = empirical_macro_drift(z, tau=cfg.tau, dim=0)
    r_drift, t_star = motion_drift_heatmap(z, tau=cfg.tau)
    r_field, _ = motion_field_heatmap(z, predictor, cond, tau=cfg.tau, t_star=t_star)
    return {
        "macro_drift_norm": float(drift.norm().item()),
        "t_star": t_star,
        "drift_heatmap_peak": float(r_drift.max().item()),
        "field_heatmap_peak": float(r_field.max().item()),
        "guidance_loss_demo": float(
            guidance_loss(
                z,
                torch.stack([predictor(z[i].unsqueeze(0), cond).squeeze(0) for i in range(z.size(0) - cfg.tau)]),
                tau=cfg.tau,
            ).item()
        ),
    }


def build_predictor(z_sample: Tensor, *, cfg: LaMoConfig | None = None) -> MotionFieldPredictor:
    """Factory for f_φ sized to latent channels."""
    cfg = cfg or LaMoConfig()
    c = z_sample.size(-3) if z_sample.dim() >= 3 else z_sample.size(0)
    return MotionFieldPredictor(
        in_channels=int(c),
        cond_dim=cfg.cond_dim,
        hidden_channels=cfg.predictor_channels,
        num_blocks=cfg.predictor_blocks,
    )
