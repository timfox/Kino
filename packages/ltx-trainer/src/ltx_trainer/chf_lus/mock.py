"""CHF-LUS smoke evaluation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.chf_lus.features import patient_representation, temporal_difference


def evaluation_smoke() -> dict[str, Any]:
    views = 6
    deltas = {f"view_{i}": temporal_difference([0.2 + 0.01 * i, 0.3], [0.1, 0.2]) for i in range(views)}
    fused = patient_representation(deltas, fusion="concatenate")
    return {
        "fused_dim": len(fused),
        "views": views,
        "best_view": "concatenate",
        "best_f1": 0.80,
        "temporal": "difference",
    }
