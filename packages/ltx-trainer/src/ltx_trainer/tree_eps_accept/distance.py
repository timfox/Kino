"""Bisimulation distance on finite trees (Sec. 3.1, 3.4)."""

from __future__ import annotations

from ltx_trainer.tree_eps_accept.trees import LabelledTree, default_lift_fn, lift_distance


def bisimulation_distance(
    t: LabelledTree,
    t_prime: LabelledTree,
    *,
    lift_name: str = "arithmetic_mean",
    max_iter: int = 64,
) -> dict[tuple[str, str], float]:
    """Greatest fixpoint bd on shared paths (finite iteration)."""
    paths_t = set(t.paths())
    paths_tp = set(t_prime.paths())
    pairs = [(w, wp) for w in paths_t for wp in paths_tp]
    f = default_lift_fn(lift_name)
    d = {(w, wp): 1.0 for w, wp in pairs}

    for _ in range(max_iter):
        changed = False
        new_d: dict[tuple[str, str], float] = {}
        for w, wp in pairs:
            if t.labels.get(w) != t_prime.labels.get(wp):
                val = 1.0
            elif w not in paths_t or wp not in paths_tp:
                val = 1.0
            else:
                val = lift_distance(d, t.gamma(w), t_prime.gamma(wp), f=f)
            if abs(val - d[(w, wp)]) > 1e-9:
                changed = True
            new_d[(w, wp)] = val
        d = new_d
        if not changed:
            break
    return d


def bd_at_roots(t: LabelledTree, t_prime: LabelledTree, **kwargs: object) -> float:
    d = bisimulation_distance(t, t_prime, **kwargs)  # type: ignore[arg-type]
    return d.get(("", ""), 1.0)
