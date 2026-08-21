"""Training and evaluation glue for group-action world models (Fig. 2–3)."""

from __future__ import annotations

import random
from typing import Literal

import torch
from torch import Tensor

from ltx_trainer.group_action.config import GroupActionConfig
from ltx_trainer.group_action.losses import group_action_loss
from ltx_trainer.group_action.metrics import gac_aggregate, gar_error
from ltx_trainer.group_action.se2 import Action, actions_to_tensor
from ltx_trainer.group_action.state import State
from ltx_trainer.group_action.synthesis import ConstraintKind, sample_constraint_kind, synthesize_constraint

ConstraintSample = Literal["id", "inv", "comp"]


def latent_endpoint(
    z_start: Tensor,
    actions: list[Action],
    transition: callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Free-running latent rollout proxy: z_{t+1} = transition(z_t, a_t)."""
    z = z_start
    for a in actions:
        a_t = actions_to_tensor([a])
        z = transition(z, a_t.squeeze(0))
    return z


def ga_training_step(
    z_start: Tensor,
    native_actions: list[Action],
    transition: callable[[Tensor, Tensor], Tensor],
    *,
    constraint: ConstraintSample | None = None,
    cfg: GroupActionConfig | None = None,
) -> dict[str, Tensor | float]:
    """One stochastic GA constraint sample (Appendix B.2)."""
    cfg = cfg or GroupActionConfig()
    kind: ConstraintSample = constraint or sample_constraint_kind()  # type: ignore[assignment]
    primary, secondary = synthesize_constraint(native_actions, kind, cfg=cfg)

    if kind == "id":
        z_end = latent_endpoint(z_start, primary, transition)
        loss = group_action_loss(z_start=z_start, z_id=z_end, lambda_id=cfg.lambda_id)
    elif kind == "inv":
        z_end = latent_endpoint(z_start, primary, transition)
        loss = group_action_loss(z_start=z_start, z_inv=z_end, lambda_inv=cfg.lambda_inv)
    else:
        assert secondary is not None
        z_a = latent_endpoint(z_start, primary, transition)
        z_b = latent_endpoint(z_start, secondary, transition)
        loss = group_action_loss(
            z_start=z_start,
            z_comp_a=z_a,
            z_comp_b=z_b,
            lambda_comp=cfg.lambda_comp,
        )
    return {
        "loss_ga": loss * cfg.lambda_ga,
        "constraint": kind,
        "loss_raw": loss.detach(),
    }


def evaluate_gac_from_probes(
    *,
    delta_id: float,
    delta_inv: float,
    delta_comp: float,
) -> dict[str, float]:
    return gac_aggregate(delta_id, delta_inv, delta_comp)


def evaluate_gar_from_rollouts(
    rollouts: list[list[State]],
    *,
    cfg: GroupActionConfig | None = None,
) -> float:
    cfg = cfg or GroupActionConfig()
    return gar_error(rollouts, alpha=cfg.rotation_weight, cfg=cfg)


def demo_identity_drift(
    states_start: list[State],
    states_after_zero: list[State],
    *,
    cfg: GroupActionConfig | None = None,
) -> float:
    """Helper: mean identity probe error for tests."""
    from ltx_trainer.group_action.metrics import identity_probe_error

    cfg = cfg or GroupActionConfig()
    return identity_probe_error(states_start, states_after_zero, alpha=cfg.rotation_weight)
