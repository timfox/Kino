"""Formal results cards — Sec. II."""

from __future__ import annotations

from typing import Any

from ltx_trainer.branch_energy.constants import THEOREMS


def theorems_card() -> dict[str, Any]:
    return {
        "theorem_1": "Branch-Level Localization: unique p_R, W'_L, W'_C given admissible T",
        "theorem_2": "Topology Indeterminacy: distinct topologies, identical terminals",
        "theorem_3": "Generalized Energetic Duality: invariant branch energy across families",
        "corollary_1": "Localization is a property of the model, not solely of data",
        "corollary_2": "Series–parallel branch energy invariant under Eq. (8)–(10) map",
        "statements": list(THEOREMS),
    }
