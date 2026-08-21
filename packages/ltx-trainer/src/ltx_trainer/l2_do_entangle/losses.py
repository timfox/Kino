"""Single-output and dual-output hybrid CTC-attention loss stubs (arXiv:2606.06065)."""

from __future__ import annotations


def single_output_loss(ctc: float, att: float, *, alpha: float = 0.2) -> float:
    """§2.2 Eq. (1): Lsingle = α LCTC + (1 − α) Latt."""
    return alpha * ctc + (1.0 - alpha) * att


def dual_output_loss(
    ctc: float,
    surf: float,
    mean: float,
    *,
    alpha: float = 0.2,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> float:
    """§2.2 Eq. (2): Ldual = α LCTC + β Lsurf + γ Lmean."""
    return alpha * ctc + beta * surf + gamma * mean


def cer_gap(do_cer: float, so_cer: float) -> float:
    """Δ = DO − SO; positive = MTL degradation, negative = improvement."""
    return round(do_cer - so_cer, 2)
