"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gcp_ism.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.04358"
    assert fw["complexity"]["gcp_ism"] == "O(N k^2 log k)"
    assert demo["fig9"]["nmse_drop_per_lambda_doubling_db"] == 12.0
    assert demo["fig5"]["ell"] == [5, 4, 3]
    return {
        "status": "ok",
        "paper": fw["paper"],
        "github": fw["github"],
        "complexity": fw["complexity"]["gcp_ism"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
