"""Training objective weights — Sec. 2.2, Eq. (3)."""

from __future__ import annotations


def total_objective(
    l_ntp: float,
    l_visual: float,
    l_dit: float,
    *,
    lambda_text: float = 0.2,
    lambda_visual: float = 1.0,
    lambda_dit: float = 1.0,
) -> float:
    r"""L = λ_text L_ntp + λ_visual L_visual + λ_dit L_dit (Eq. 3)."""
    return lambda_text * l_ntp + lambda_visual * l_visual + lambda_dit * l_dit
