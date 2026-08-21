"""Deadlock-free tree automata toys (Sec. 2.2, 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.tree_eps_accept.trees import LabelledTree, Successor


@dataclass(frozen=True)
class TreeAutomaton:
    states: frozenset[str]
    initial: str
    transitions: dict[str, frozenset[Successor]]
    name: str = "A"

    def choices(self, a: str) -> frozenset[Successor]:
        return self.transitions.get(a, frozenset())


def termination_automaton(*, branch: str = "σ", leaf: str = "★") -> TreeAutomaton:
    """Sec. 5: branch (σ, a0, a0) on inner nodes; leaf symbol on Σ0 nodes."""
    a0 = "a0"
    return TreeAutomaton(
        states=frozenset({a0}),
        initial=a0,
        transitions={
            a0: frozenset({leaf, (branch, a0, a0)}),
        },
        name="termination",
    )


def no_error_automaton(*, ok: str = "ok", leaf: str = "★") -> TreeAutomaton:
    """Sec. 5: Δ(a0) = {★, (ok, a0, a0)}."""
    a0 = "a0"
    return TreeAutomaton(
        states=frozenset({a0}),
        initial=a0,
        transitions={
            a0: frozenset({leaf, (ok, a0, a0)}),
        },
        name="no_error",
    )


def accepts_position(
    auto: TreeAutomaton,
    a: str,
    w: str,
    tree: LabelledTree,
    *,
    memo: dict[tuple[str, str], bool] | None = None,
) -> bool:
    """Rigid acceptance: ∃ winning strategy in Table 1 (finite unfolding)."""
    memo = memo or {}
    key = (a, w)
    if key in memo:
        return memo[key]
    if w not in tree.labels:
        memo[key] = False
        return False
    succ = tree.gamma(w)
    for delta in auto.choices(a):
        if not _delta_matches_tree(delta, succ):
            continue
        pairs = _successor_pairs(delta, succ)
        if not pairs:
            memo[key] = True
            return True
        if all(accepts_position(auto, b, v, tree, memo=memo) for b, v in pairs):
            memo[key] = True
            return True
    memo[key] = False
    return False


def _delta_matches_tree(delta: Successor, succ: Successor) -> bool:
    if isinstance(delta, str) and isinstance(succ, str):
        return delta == succ
    if isinstance(delta, tuple) and isinstance(succ, tuple):
        return delta[0] == succ[0]
    return False


def _successor_pairs(delta: Successor, succ: Successor) -> list[tuple[str, str]]:
    if isinstance(delta, str):
        return []
    if isinstance(succ, str):
        return []
    (_, b0, b1), (_, v0, v1) = delta, succ
    return [(b0, v0), (b1, v1)]
