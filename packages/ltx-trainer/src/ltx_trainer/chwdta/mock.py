"""CPU smoke for paper stub."""

from __future__ import annotations

from typing import Any

from ltx_trainer.chwdta.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.00111"
    assert fw["headline_bd_rate_8"]["Kodak"] == -17.82
    assert demo["table2_ablation"][0]["bd_rate"] == -17.82
    assert demo["chwp"]["num_slices_8"] == 8
    return {
        "status": "ok",
        "paper": fw["paper"],
        "bd_kodak_8": fw["headline_bd_rate_8"]["Kodak"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
