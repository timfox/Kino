"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hybrid_aver.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.03747"
    assert fw["dataset"] == "AVE"
    hybrid = next(r for r in demo["table1"] if "Hybrid" in r["model"])
    assert hybrid["test_acc"] == "0.8385 ± 0.0140"
    t2 = next(r for r in demo["table2"] if "Hybrid" in r["model"])
    assert t2["params_M"] == 6.856
    return {
        "status": "ok",
        "paper": fw["paper"],
        "test_acc": demo["framework"]["headline"]["test_acc"],
        "params_M": t2["params_M"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
