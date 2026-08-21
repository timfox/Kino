"""Labelled binary trees and successor maps (Sec. 2.1–2.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Sigma0 = str
Sigma2 = str
Successor = Sigma0 | tuple[Sigma2, str, str]


def child_path(w: str, i: int) -> str:
    """Left-child / right-sibling style paths: root children are ``0``, ``1``."""
    return f"{w}.{i}" if w else str(i)


@dataclass(frozen=True)
class LabelledTree:
    """Prefix- and sibling-closed tree as path → label (Σ0 ∪ Σ2)."""

    labels: dict[str, str]
    sigma0: frozenset[str]
    sigma2: frozenset[str]

    def __post_init__(self) -> None:
        if "" not in self.labels:
            raise ValueError("root path '' required")
        for w, lab in self.labels.items():
            c0, c1 = child_path(w, 0), child_path(w, 1)
            if lab in self.sigma0:
                if c0 in self.labels or c1 in self.labels:
                    raise ValueError(f"leaf {w} cannot have children")
            elif lab in self.sigma2:
                if c0 not in self.labels or c1 not in self.labels:
                    raise ValueError(f"branch {w} requires both children")
            else:
                raise ValueError(f"unknown label {lab!r} at {w}")

    def arity(self, w: str) -> int:
        return 0 if self.labels[w] in self.sigma0 else 2

    def gamma(self, w: str) -> Successor:
        lab = self.labels[w]
        if self.arity(w) == 0:
            return lab
        return (lab, child_path(w, 0), child_path(w, 1))

    def paths(self) -> list[str]:
        return sorted(self.labels.keys(), key=lambda p: (len(p), p))

    def leaves(self) -> list[str]:
        return [w for w in self.paths() if self.arity(w) == 0]


def full_binary_tree(depth: int, *, branch: str = "σ", leaf: str = "★") -> LabelledTree:
    """Full binary tree of uniform depth (paper Claim 5.2 TA)."""
    labels: dict[str, str] = {}

    def fill(w: str, d: int) -> None:
        if d == 0:
            labels[w] = leaf
            return
        labels[w] = branch
        fill(child_path(w, 0), d - 1)
        fill(child_path(w, 1), d - 1)

    fill("", depth)
    return LabelledTree(labels=labels, sigma0=frozenset({leaf}), sigma2=frozenset({branch}))


def tree_with_leaves_at_depth_one(*, branch: str = "σ", leaf: str = "★") -> LabelledTree:
    """Root branch; both children leaves (leaf mass = 1 under μ)."""
    return LabelledTree(
        labels={"": branch, "0": leaf, "1": leaf},
        sigma0=frozenset({leaf}),
        sigma2=frozenset({branch}),
    )


def single_leaf_chain(depth: int, *, branch: str = "σ", leaf: str = "★") -> LabelledTree:
    """Left-spine branch chain ending in one leaf at depth `depth`."""
    labels: dict[str, str] = {}
    w = ""
    for _ in range(depth):
        labels[w] = branch
        w = child_path(w, 0)
    labels[w] = leaf
    return LabelledTree(labels=labels, sigma0=frozenset({leaf}), sigma2=frozenset({branch}))


def tree_with_error_subtree(*, ok: str = "ok", error: str = "error", leaf: str = "★") -> LabelledTree:
    """Root ok; left ok branch to leaves; right error root (Sec. 5 failed executions)."""
    return LabelledTree(
        labels={
            "": ok,
            "0": ok,
            "0.0": leaf,
            "0.1": leaf,
            "1": error,
            "1.0": leaf,
            "1.1": leaf,
        },
        sigma0=frozenset({leaf}),
        sigma2=frozenset({ok, error}),
    )


LiftFn = Callable[[float, float], float]


def default_lift_fn(name: str = "arithmetic_mean") -> LiftFn:
    if name == "arithmetic_mean":
        return lambda x, y: 0.5 * x + 0.5 * y
    if name == "max":
        return lambda x, y: max(x, y)
    raise ValueError(f"unknown lift {name!r}")


def lift_distance(
    d: dict[tuple[str, str], float],
    alpha: Successor,
    beta: Successor,
    *,
    f: LiftFn | None = None,
) -> float:
    """Distance lifting d̄ (Def. 3.1)."""
    f = f or default_lift_fn()
    if isinstance(alpha, str) and isinstance(beta, str):
        return 0.0 if alpha == beta else 1.0
    if isinstance(alpha, str) or isinstance(beta, str):
        return 1.0
    (_, w1, w2), (_, w1p, w2p) = alpha, beta
    if alpha[0] != beta[0]:
        return 1.0
    return f(d.get((w1, w1p), 1.0), d.get((w2, w2p), 1.0))


def lift_relation(
    pairs: set[tuple[str, str]],
    alpha: Successor,
    beta: Successor,
) -> bool:
    """Relation lifting R̄ (Def. 2.6)."""
    if isinstance(alpha, str) and isinstance(beta, str):
        return alpha == beta
    if isinstance(alpha, str) or isinstance(beta, str):
        return False
    (_, w1, w2), (_, w1p, w2p) = alpha, beta
    if alpha[0] != beta[0]:
        return False
    return (w1, w1p) in pairs and (w2, w2p) in pairs
