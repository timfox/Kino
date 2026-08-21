"""Game tables and paper headlines (arXiv:2605.27192)."""

from __future__ import annotations

from typing import Any


def acceptance_game_table() -> list[dict[str, str]]:
    """Table 1 — rigid acceptance game [15]."""
    return [
        {"position": "(a, w) ∈ A × U", "player": "∃", "moves": "δ(a) ∈ ∆(a)"},
        {"position": "(δ(a), w)", "player": "∃", "moves": "R ⊆ A×U with (δ(a),γ(w)) ∈ R̄"},
        {"position": "R", "player": "∀", "moves": "(b, v) ∈ R"},
    ]


def bisimulation_distance_game_table() -> list[dict[str, str]]:
    """Table 2."""
    return [
        {"position": "(w, w′, ε1)", "player": "∃", "moves": "d : U×U′→[0,1], d̄(γ(w),γ′(w′))≤ε1"},
        {"position": "d", "player": "∀", "moves": "(v, v′, ε2) with d(v,v′)≤ε2"},
    ]


def epsilon_acceptance_game_table() -> list[dict[str, str]]:
    """Table 3."""
    return [
        {"position": "(a, w, ε1)", "player": "∃", "moves": "(δ(a), w, ε1), δ(a)∈∆(a)"},
        {"position": "(δ(a), w, ε1)", "player": "∃", "moves": "d : A×U→[0,1], d̄(δ(a),γ(w))≤ε1"},
        {"position": "d", "player": "∀", "moves": "(b, v, ε2) with d(b,v)≤ε2"},
    ]


def headline_results() -> dict[str, Any]:
    return {
        "main_theorem": "Thm. 4.1: ε-accepted ⇔ ∃ rigidly accepted T′ with bd(T′,T)≤ε",
        "bisimulation_game_theorem": "Thm. 3.7: (w,w′,ε)∈Win_bd_∃ ⇔ bd(Tw,T′_w′)≤ε",
        "prop_5_1": "termination: ε-acceptance ⇔ ε ≥ Σ_{l∈L} μ(l|root)",
        "prop_5_3": "no error: ε-acceptance ⇔ ε ≥ μ(dTerr)",
        "distance_lifting": "coupling-based lifting with order-preserving f : [0,1]²→[0,1]",
        "functor": "F X = Σ0 ⊎ (Σ2 × X × X)",
    }
