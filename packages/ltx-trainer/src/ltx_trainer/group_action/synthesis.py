"""Group-action supervision synthesis (Sec. 3.2, Appendix B)."""

from __future__ import annotations

import random
from typing import Literal

import torch

from ltx_trainer.group_action.config import GroupActionConfig
from ltx_trainer.group_action.se2 import (
    Action,
    composition_alternative_segment,
    inverse_segment,
    zero_action,
)

ConstraintKind = Literal["id", "inv", "comp"]


def sample_horizon(cfg: GroupActionConfig | None = None) -> int:
    cfg = cfg or GroupActionConfig()
    return random.randint(1, cfg.max_rollout_horizon)


def identity_segment(length: int) -> list[Action]:
    """Eq. (15): zero-action segment."""
    return [zero_action() for _ in range(length)]


def sample_dirichlet_weights(length: int, *, cfg: GroupActionConfig | None = None) -> list[float]:
    cfg = cfg or GroupActionConfig()
    w = torch.distributions.Dirichlet(torch.ones(length) * cfg.dirichlet_alpha).sample()
    return w.tolist()


def synthesize_constraint(
    native_segment: list[Action],
    kind: ConstraintKind,
    *,
    cfg: GroupActionConfig | None = None,
) -> tuple[list[Action], list[Action] | None]:
    """Return (primary_segment, optional_second_segment) for composition."""
    cfg = cfg or GroupActionConfig()
    if kind == "id":
        l = len(native_segment) or sample_horizon(cfg)
        return identity_segment(l), None
    if kind == "inv":
        seg = native_segment or [zero_action()]
        return inverse_segment(seg), None
    if kind == "comp":
        seg = native_segment or [zero_action() for _ in range(sample_horizon(cfg))]
        w = sample_dirichlet_weights(len(seg), cfg=cfg)
        alt = composition_alternative_segment(seg, weights=w)
        return seg, alt
    raise ValueError(f"unknown constraint kind: {kind}")


def sample_constraint_kind() -> ConstraintKind:
    return random.choice(["id", "inv", "comp"])
