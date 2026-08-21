"""Theoretical motivation stubs — Theorems 4.4–4.5."""

from __future__ import annotations

import math
from typing import Any


def shell_confinement_rate(T: int, epsilon: float = 0.05) -> float:
    """Pr[Y ∉ A^ε_T] ≤ exp(-Ω(T)) — Theorem 4.4a (qualitative)."""
    return math.exp(-0.01 * T * max(epsilon, 1e-6))


def evolution_escape_fraction(gamma: float, epsilon: float, L: float, H_T: float, T: int) -> float:
    """Upper bound on Pr_Q[Ỹ ∈ A^ε_T] — Eq. (9), Theorem 4.4b."""
    num = (gamma - epsilon) * T
    den = L * T - H_T - epsilon * T
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - num / den))


def sample_complexity_ratio(
    subgoal_probs: list[float], *, delta: float = 0.05
) -> dict[str, Any]:
    """
    Theorem 4.5 — terminal-only vs bidirectional sample counts.

    N_term = Ω(1 / ∏ p_i); N_bidir = O(p_min^{-1} log(m/δ)).
    """
    m = len(subgoal_probs)
    if m == 0:
        return {"n_term": 0, "n_bidir": 0, "ratio": 1.0}
    prod = 1.0
    for p in subgoal_probs:
        prod *= max(p, 1e-12)
    p_min = min(subgoal_probs)
    n_term = 1.0 / prod
    n_bidir = math.log(m / max(delta, 1e-12)) / max(p_min, 1e-12)
    ratio = n_term / max(n_bidir, 1e-12)
    return {
        "m_subgoals": m,
        "p_product": prod,
        "p_min": p_min,
        "n_term_order": n_term,
        "n_bidir_order": n_bidir,
        "ratio_n_term_over_n_bidir": ratio,
        "exponential_in_m": m > 1 and ratio > 10,
    }


def theory_card() -> dict[str, Any]:
    """Summary for framework / eval demos."""
    sym = sample_complexity_ratio([0.2] * 5)
    return {
        "theorem_4_4a": "Expansion-only candidates confined to narrow entropy shell",
        "theorem_4_4b": "Evolution operators escape shell via block total correlation γ",
        "theorem_4_5": "Backward sub-goals reduce required samples exponentially in m",
        "symmetric_p0.2_m5": sym,
        "shell_escape_gamma_0.1": evolution_escape_fraction(0.1, 0.05, 10.0, 5.0, 100),
    }
