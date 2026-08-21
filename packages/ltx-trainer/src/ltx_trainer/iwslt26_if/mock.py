"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.iwslt26_if.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    t2 = demo["table2_primary"]
    t3 = {r["method"]: r for r in demo["table3_rerank"]}
    assert fw["paper"] == "arXiv:2606.04730"
    assert fw["training"]["samples"] == 1_048_158
    assert t2["ASR_fix"] == 37.65
    assert t3["Lik. + MBR"]["impr"] == 3.71
    return {
        "status": "ok",
        "paper": fw["paper"],
        "samples": fw["training"]["samples"],
        "lik_mbr_impr": t3["Lik. + MBR"]["impr"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
