"""GAC and GAR metrics (Sec. 3.3, Eq. 19–21)."""

from __future__ import annotations

from typing import Sequence

from ltx_trainer.group_action.config import GroupActionConfig
from ltx_trainer.group_action.state import State, state_distance, trajectory_gar_distance


def identity_probe_error(
    states_before: Sequence[State],
    states_after: Sequence[State],
    *,
    alpha: float = 1.0,
) -> float:
    """Mean d(s_after, s_before) over probe segments (Eq. 42)."""
    if not states_before or len(states_before) != len(states_after):
        return 0.0
    errs = [state_distance(sa, sb, alpha=alpha) for sb, sa in zip(states_before, states_after)]
    return sum(errs) / len(errs)


def inverse_probe_error(
    states_start: Sequence[State],
    states_recovered: Sequence[State],
    *,
    alpha: float = 1.0,
) -> float:
    """Eq. (43): recovery after forward–inverse."""
    if not states_start:
        return 0.0
    errs = [state_distance(sr, ss, alpha=alpha) for sr, ss in zip(states_recovered, states_start)]
    return sum(errs) / len(errs)


def composition_probe_error(
    endpoints_a: Sequence[State],
    endpoints_b: Sequence[State],
    *,
    alpha: float = 1.0,
) -> float:
    """Eq. (46): endpoint mismatch under equivalent decompositions."""
    if not endpoints_a:
        return 0.0
    errs = [state_distance(ea, eb, alpha=alpha) for ea, eb in zip(endpoints_a, endpoints_b)]
    return sum(errs) / len(errs)


def gac_aggregate(
    delta_id: float,
    delta_inv: float,
    delta_comp: float,
) -> dict[str, float]:
    """Eq. (19)–(20): EGAC = mean of component errors."""
    egac = (delta_id + delta_inv + delta_comp) / 3.0
    return {
        "delta_id": delta_id,
        "delta_inv": delta_inv,
        "delta_comp": delta_comp,
        "EGAC": egac,
    }


def gar_error(
    rollouts: Sequence[Sequence[State]],
    *,
    alpha: float = 1.0,
    cfg: GroupActionConfig | None = None,
) -> float:
    """Eq. (21): pairwise rollout dispersion (lower is better)."""
    _ = cfg
    return trajectory_gar_distance(rollouts, alpha=alpha)
