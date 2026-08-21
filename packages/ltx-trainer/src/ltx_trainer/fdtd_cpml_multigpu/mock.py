"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fdtd_cpml_multigpu.paper import paper_card
from ltx_trainer.fdtd_cpml_multigpu.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "peer_speedup_ok": demo["exchange"]["speedup"] >= 2.46,
        "status": "ok",
    }
