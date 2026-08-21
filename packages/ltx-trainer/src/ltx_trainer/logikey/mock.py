"""LogiKEy evaluation smoke (arXiv:2605.27246)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.logikey.config import LogikeyConfig
from ltx_trainer.logikey.godel import (
    cantor_no_surjection_smoke,
    finite_cardinality_smoke,
    infinity_smoke,
)
from ltx_trainer.logikey.homl import KripkeFrame, boolean_extensionality_counterexample_smoke, mvalid
from ltx_trainer.logikey.tables import headline_results


def evaluation_smoke(cfg: LogikeyConfig | None = None) -> dict[str, Any]:
    c = cfg or LogikeyConfig()
    frame = KripkeFrame.s5_total(("w1", "w2", "w3"))

    def always_true(w: str) -> bool:
        return True

    return {
        "paper": c.paper_arxiv,
        "layers": list(c.layer_ids),
        "s5_frame_checks": {
            "reflexive": frame.is_reflexive(),
            "symmetric": frame.is_symmetric(),
            "transitive": frame.is_transitive(),
        },
        "mvalid_tautology": mvalid(frame, always_true),
        "extensionality_counterexample": boolean_extensionality_counterexample_smoke(),
        "godel_finite_cardinality": finite_cardinality_smoke(),
        "cantor_finite_smoke": cantor_no_surjection_smoke(3, 8),
        "godel_infinity": infinity_smoke(),
        "headline": headline_results()["cardinality"],
    }
