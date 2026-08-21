"""Training-step helpers wiring HAP pairs to Anatomical DPO (Fig. 2)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.asap.config import ASAPConfig
from ltx_trainer.asap.losses import (
    anatomical_alignment_loss,
    diffusion_dpo_loss,
    flow_matching_loss,
    preference_gap,
)
from ltx_trainer.asap.hap import HAPPair


def fm_losses_for_pair(
    v_pred_w: Tensor,
    v_pred_l: Tensor,
    v_target_w: Tensor,
    v_target_l: Tensor,
    mask: Tensor,
    *,
    cfg: ASAPConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Localized FM losses on winner/loser for one training step."""
    cfg = cfg or ASAPConfig()
    lw = flow_matching_loss(v_pred_w, v_target_w, mask, alpha=cfg.spatial_weight_alpha)
    ll = flow_matching_loss(v_pred_l, v_target_l, mask, alpha=cfg.spatial_weight_alpha)
    return lw, ll


def asap_training_loss(
    loss_policy_w: Tensor,
    loss_policy_l: Tensor,
    loss_ref_w: Tensor,
    loss_ref_l: Tensor,
    *,
    cfg: ASAPConfig | None = None,
    use_bounded: bool = True,
) -> dict[str, Tensor]:
    """ASAP vs vanilla DPO toggle for ablations (Tab. 1, Fig. 6)."""
    cfg = cfg or ASAPConfig()
    delta = preference_gap(loss_policy_w, loss_policy_l, loss_ref_w, loss_ref_l)
    out: dict[str, Tensor] = {"delta": delta.detach()}
    if use_bounded:
        loss, _ = anatomical_alignment_loss(
            loss_policy_w, loss_policy_l, loss_ref_w, loss_ref_l, tau=cfg.margin_tau
        )
        out["loss"] = loss
        out["mode"] = torch.tensor(0)  # bounded
    else:
        out["loss"] = diffusion_dpo_loss(delta, beta=cfg.dpo_beta)
        out["mode"] = torch.tensor(1)  # dpo
    return out


def synthetic_training_step(
    pair: HAPPair,
    v_policy_w: Tensor,
    v_policy_l: Tensor,
    v_ref_w: Tensor,
    v_ref_l: Tensor,
    v_target: Tensor,
    *,
    cfg: ASAPConfig | None = None,
) -> Tensor:
    """End-to-end scalar loss for one HAP pair (same rectified-flow target on both branches)."""
    cfg = cfg or ASAPConfig()
    mask = pair.mask
    if mask is None:
        mask = torch.ones_like(v_policy_w[:, :1])
    lw, ll = fm_losses_for_pair(v_policy_w, v_policy_l, v_target, v_target, mask, cfg=cfg)
    lr_w = flow_matching_loss(v_ref_w, v_target, mask, alpha=cfg.spatial_weight_alpha)
    lr_l = flow_matching_loss(v_ref_l, v_target, mask, alpha=cfg.spatial_weight_alpha)
    result = asap_training_loss(lw, ll, lr_w, lr_l, cfg=cfg, use_bounded=True)
    return result["loss"]
