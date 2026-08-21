"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.branch_energy.paper import paper_card
from ltx_trainer.branch_energy.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "status": "ok"
        if demo["max_balance_residual"] < 0.01
        and demo["open_phase_ic_zero"]
        and demo.get("open_phase_cpc_ghost", demo["cases"]["IV-B"].get("cpc_ghost_in_open_phase", False))
        and demo["triac_no_storage"]
        and demo.get("triac_pseudo_reactive", demo["cases"]["IV-C"].get("pseudo_reactive_nontrivial", False))
        and demo["duality_ok"]
        and demo["topology_profiles_differ"]
        else "fail",
    }
