"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.uat.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    t1 = demo["table1_tta"]["AudioCaps"]
    t4 = demo["table4_captioning"]
    assert fw["paper"] == "arXiv:2606.04939"
    assert fw["params_b"] == 1.7
    assert t1["IS"] >= 12.0
    assert t4["CIDEr"] >= 0.4
    return {
        "status": "ok",
        "paper": fw["paper"],
        "params_b": fw["params_b"],
        "audiocaps_IS": t1["IS"],
        "CIDEr": t4["CIDEr"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
