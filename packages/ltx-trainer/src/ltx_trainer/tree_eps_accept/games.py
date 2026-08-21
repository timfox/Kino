"""ε-acceptance and bisimulation distance games (Sec. 3.2–3.3, Tables 2–3)."""

from __future__ import annotations

from ltx_trainer.tree_eps_accept.automata import TreeAutomaton, accepts_position
from ltx_trainer.tree_eps_accept.distance import bd_at_roots
from ltx_trainer.tree_eps_accept.measure import defect_ok_mass, leaf_mass_from
from ltx_trainer.tree_eps_accept.trees import LabelledTree, default_lift_fn, lift_distance


def winning_bisimulation_distance(
    t: LabelledTree,
    t_prime: LabelledTree,
    w: str,
    wp: str,
    epsilon: float,
) -> bool:
    """(w, w′, ε) ∈ Win_bd_∃  iff bd(Tw, T′_w′) ≤ ε (Thm. 3.7 smoke)."""
    from ltx_trainer.tree_eps_accept.distance import bisimulation_distance

    d = bisimulation_distance(t, t_prime)
    return d.get((w, wp), 1.0) <= epsilon + 1e-9


def epsilon_acceptance_threshold_leaf_mass(
    tree: LabelledTree,
    epsilon: float,
    *,
    w: str = "",
) -> bool:
    """Prop. 5.1: (a0, root, ε) ∈ AWin^ε_∃ iff ε ≥ Σ_{l∈L} μ(l|root)."""
    mass = leaf_mass_from(w, tree)
    return epsilon + 1e-9 >= mass


def epsilon_acceptance_threshold_error_mass(
    tree: LabelledTree,
    epsilon: float,
) -> bool:
    """Prop. 5.3: (a0, root, ε) ∈ AWin^ε_∃ iff ε ≥ μ(dTerr)."""
    mass = defect_ok_mass(tree)
    return epsilon + 1e-9 >= mass


def theorem_4_1_direction_2(
    auto: TreeAutomaton,
    t: LabelledTree,
    t_prime: LabelledTree,
    epsilon: float,
) -> bool:
    """If T′ accepted and bd(T′,T)≤ε then ε-acceptance (Thm. 4.1 item 2 ⇒ 1 smoke)."""
    if not accepts_position(auto, auto.initial, "", t_prime):
        return False
    if bd_at_roots(t_prime, t) > epsilon + 1e-9:
        return False
    # Constructive witness: use d from bd game — smoke via leaf/error thresholds
    if auto.name == "termination":
        return epsilon_acceptance_threshold_leaf_mass(t, epsilon)
    if auto.name == "no_error":
        return epsilon_acceptance_threshold_error_mass(t, epsilon)
    return bd_at_roots(t_prime, t) <= epsilon + 1e-9


def verifier_move_valid(
    delta_succ: tuple[object, str],
    tree: LabelledTree,
    w: str,
    d: dict[tuple[str, str], float],
    epsilon: float,
) -> bool:
    """¯d(δ(a), γ(w)) ≤ ε for Table 3 middle row."""
    delta, _ = delta_succ
    from ltx_trainer.tree_eps_accept.trees import Successor

    succ = tree.gamma(w)
    if not isinstance(delta, Successor):
        return False
    f = default_lift_fn()
    return lift_distance(d, delta, succ, f=f) <= epsilon + 1e-9
