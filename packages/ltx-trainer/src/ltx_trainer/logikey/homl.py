"""Toy HOML-as-HOL shallow embedding (world-indexed predicates)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class KripkeFrame:
    """Finite Kripke frame for smoke tests."""

    worlds: tuple[str, ...]
    accessible: frozenset[tuple[str, str]]  # (w, v) ∈ R

    @classmethod
    def s5_total(cls, worlds: tuple[str, ...]) -> KripkeFrame:
        edges = tuple((w, v) for w in worlds for v in worlds)
        return cls(worlds=worlds, accessible=frozenset(edges))

    def successors(self, w: str) -> tuple[str, ...]:
        return tuple(v for (a, v) in self.accessible if a == w)

    def is_reflexive(self) -> bool:
        return all((w, w) in self.accessible for w in self.worlds)

    def is_symmetric(self) -> bool:
        return all(
            (v, w) in self.accessible for (w, v) in self.accessible
        )

    def is_transitive(self) -> bool:
        for w in self.worlds:
            for v in self.successors(w):
                for u in self.successors(v):
                    if (w, u) not in self.accessible:
                        return False
        return True


Prop = Callable[[str], bool]  # world → bool


def mvalid(frame: KripkeFrame, phi: Prop) -> bool:
    """Global validity: ∀w. φ w."""
    return all(phi(w) for w in frame.worlds)


def box(frame: KripkeFrame, w: str, phi: Prop) -> bool:
    """□φ at w: φ holds at all R-successors of w."""
    succ = frame.successors(w)
    return bool(succ) and all(phi(v) for v in succ)


def diamond(frame: KripkeFrame, w: str, phi: Prop) -> bool:
    """◇φ at w: φ at some successor."""
    return any(phi(v) for v in frame.successors(w))


def leibniz_equiv_at(w: str, phi_a: Prop, phi_b: Prop) -> bool:
    """φ ↔ ψ at world w (lifted equivalence for extensionality discussion)."""
    return phi_a(w) == phi_b(w)


def hol_identity(frame: KripkeFrame, phi_a: Prop, phi_b: Prop) -> bool:
    """HOL identity on σ: agreement at every world in the frame."""
    return all(phi_a(w) == phi_b(w) for w in frame.worlds)


def boolean_extensionality_counterexample_smoke() -> dict[str, object]:
    """
    Nitpick-style two-world countermodel (paper §3.2 footnote):
    φ false everywhere, ψ false at i1 and true at i2; at w=i1 lifted equiv holds but φ≠ψ.
    """
    frame = KripkeFrame(
        worlds=("i1", "i2"),
        accessible=frozenset({("i1", "i1"), ("i1", "i2"), ("i2", "i1"), ("i2", "i2")}),
    )

    def phi(w: str) -> bool:
        return False

    def psi(w: str) -> bool:
        return w == "i2"

    w = "i1"
    lifted = leibniz_equiv_at(w, phi, psi)
    global_neq = not hol_identity(frame, phi, psi)
    one_world_restores = frame.worlds == ("i1",) and leibniz_equiv_at("i1", phi, psi)

    return {
        "worlds": frame.worlds,
        "lifted_equiv_at_i1": lifted,
        "hol_global_identity": not global_neq,
        "extensionality_blocked_without_one_world": lifted and global_neq,
        "one_world_collapses_distinction": one_world_restores,
    }
