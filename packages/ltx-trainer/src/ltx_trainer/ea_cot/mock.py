"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ea_cot.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    recovery = demo["wol_recovery_pp"]
    ablation = {r["component"]: r for r in demo["table2_ablation"]}
    assert fw["paper"] == "arXiv:2606.04474"
    assert recovery["Phi-4-MM"] == 24.4
    assert ablation["Full EA-CoT"]["delta_pp"] == 17.6
    assert abs(demo["token_budget_speech"]["instruction_delta_pp"] - 16.8) < 0.05
    return {
        "status": "ok",
        "paper": fw["paper"],
        "phi_wol_gain_pp": recovery["Phi-4-MM"],
        "full_ea_cot_delta_pp": ablation["Full EA-CoT"]["delta_pp"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
