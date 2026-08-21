"""CAST losses: step KL + operator regularizer (Appendix D.5)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cast.config import CASTConfig, SMOOTHNESS_WEIGHT
from ltx_trainer.cast.simplex import kl_divergence, normalize_simplex, support_mean


def step_kl_loss(target: Tensor, pred: Tensor) -> Tensor:
    return kl_divergence(normalize_simplex(target), normalize_simplex(pred)).mean()


def operator_regularizer(
    rho: Tensor,
    kernel_logits: Tensor,
    anchor: Tensor,
    transported: Tensor,
    *,
    cfg: CASTConfig,
) -> Tensor:
    """Target-free R_op on transport strength, off-identity mass, smoothness, mean shift."""
    # strength penalty
    strength = rho.mean()
    probs = torch.softmax(kernel_logits, dim=-1)
    center = probs.shape[-1] // 2
    off_identity = (probs[..., :center].sum(-1) + probs[..., center + 1 :].sum(-1)).pow(2).mean()
    # neighbor smoothness
    if probs.shape[-1] > 1:
        smooth = (probs[..., 1:, :] - probs[..., :-1, :]).pow(2).mean()
    else:
        smooth = torch.tensor(0.0, device=anchor.device)
    delta_mu = support_mean(transported) - support_mean(anchor)
    support_span = anchor.argmax(dim=-1).float() - anchor.argmin(dim=-1).float()
    budget = cfg.delta_mu + cfg.delta_sigma * support_span.abs()
    mean_shift = ((delta_mu / (budget + 1e-6)).pow(2)).mean()
    return strength + off_identity + SMOOTHNESS_WEIGHT * smooth + mean_shift


def cast_loss(
    target: Tensor,
    pred: Tensor,
    *,
    rho: Tensor,
    kernel_logits: Tensor,
    anchor: Tensor,
    transported: Tensor,
    cfg: CASTConfig,
) -> tuple[Tensor, dict[str, float]]:
    kl = step_kl_loss(target, pred)
    rop = operator_regularizer(rho, kernel_logits, anchor, transported, cfg=cfg)
    loss = kl + cfg.lambda_op * rop
    return loss, {"loss_kl": float(kl.detach()), "loss_op": float(rop.detach()), "loss": float(loss.detach())}
