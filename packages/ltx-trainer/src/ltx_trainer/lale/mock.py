"""CPU smoke for paper stub validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lale.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.02092"
    assert fw["headline"]["f1_gap"] == 2.6
    s1 = demo["forward_s1"]
    assert s1["logits_shape"] == [2, 8, 256, 256]
    assert s1["params_m"] > 0.1
    lale_rows = [r for r in demo["table1_aras400k"] if r["architecture"].startswith("LALE")]
    assert len(lale_rows) >= 2
    assert lale_rows[0]["throughput"] > 100000
    return {
        "status": "ok",
        "paper": fw["paper"],
        "lale_s1_params_m": s1["params_m"],
        "best_ablation": "S2-K3-PT",
    }


__all__ = ["evaluation_smoke", "framework_card"]
