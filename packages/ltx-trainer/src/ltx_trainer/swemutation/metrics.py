"""Evaluation metrics: Pass@1, VRR, RDR (Sec. 4.3, Eq. 1)."""

from __future__ import annotations

from typing import Iterable


def pass_at_1(successes: int, total: int) -> float:
    """Fraction of instances with valid, applicable test diffs."""
    if total <= 0:
        return 0.0
    return float(successes / total * 100.0)


def verified_reproduction_rate(
    reproduced_and_valid: int,
    total: int,
) -> float:
    """VRR: fails on bug AND passes on golden (Sec. 4.3)."""
    if total <= 0:
        return 0.0
    return float(reproduced_and_valid / total * 100.0)


def relative_detection_rate(
    killed_by_gen: Iterable[int],
    killed_by_base: Iterable[int],
    total_mutants: Iterable[int],
) -> float:
    """RDR micro-average (Eq. 1).

    RDR = Σ|M_gen \\ M_base| / Σ|M \\ M_base|
    """
    num = 0
    den = 0
    for m_gen, m_base, m_all in zip(killed_by_gen, killed_by_base, total_mutants, strict=True):
        survivors = set(range(m_all)) - set(m_base)
        newly_killed = set(m_gen) & survivors
        num += len(newly_killed)
        den += len(survivors)
    if den <= 0:
        # test generation: M_base = ∅ → absolute mutation score
        total_killed = sum(len(set(g)) for g in killed_by_gen)
        total_m = sum(total_mutants)
        return float(total_killed / total_m * 100.0) if total_m else 0.0
    return float(num / den * 100.0)


def absolute_mutation_score(killed: int, total: int) -> float:
    """|M_gen| / |M| when M_base = ∅ (test generation)."""
    if total <= 0:
        return 0.0
    return float(killed / total * 100.0)
