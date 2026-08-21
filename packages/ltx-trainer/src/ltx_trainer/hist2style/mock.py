"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hist2style.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.01819"
    assert fw["params_m"] == 1.5
    assert demo["table1_user_study"][0]["h2s_win"] > 61.0
    return {"status": "ok", "paper": fw["paper"], "params_m": fw["params_m"]}


__all__ = ["evaluation_smoke", "framework_card"]
