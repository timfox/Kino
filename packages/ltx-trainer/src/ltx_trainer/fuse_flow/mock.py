"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fuse_flow.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2602.01035"
    assert fw["decoupled"] is True
    gmac = next(r for r in demo["table1_gmac_scannet"] if r["method"] == "GMAC")
    assert gmac["rot_error_deg"] == 0.63
    fuse = next(r for r in demo["table4_fuse_static"] if r["method"] == "FUSE")
    assert fuse["pfr_mm"] == 8.79
    return {
        "status": "ok",
        "paper": fw["paper"],
        "title": fw["title"],
        "modules": list(fw["modules"]),
    }


__all__ = ["evaluation_smoke", "framework_card"]
