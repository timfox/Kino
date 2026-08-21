"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.paper import paper_card
from ltx_trainer.deopt_reopt.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    conv = demo["conv2d_highlight"]
    return {
        "paper": paper_card(),
        "demo": demo,
        "conv2d_dr_wins_both_models": all(
            conv["single_shot"][m]["bh_significant"] for m in ("O120", "Q235")
        ),
        "bh_claims_consistent": demo["all_bh_claims_consistent"],
        "status": "ok" if demo["all_bh_claims_consistent"] else "fail",
    }
