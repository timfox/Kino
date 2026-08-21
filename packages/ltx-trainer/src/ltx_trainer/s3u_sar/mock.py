"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3u_sar.paper import paper_card
from ltx_trainer.s3u_sar.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "ap_anchor": demo["summary"]["AP"],
        "status": "ok" if demo["beats_hrnet_w32"] and demo["summary"]["AP"] >= 59.0 else "fail",
    }
