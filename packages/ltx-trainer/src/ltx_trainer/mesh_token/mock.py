"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mesh_token.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.02000"
    assert fw["representation"]["render_free"] is True
    t1 = demo["table1_trajectory100"][-1]
    assert t1["method"] == "MeshToken (ours)"
    assert t1["psnr"] == 16.78
    assert demo["forward"]["motion_tokens_shape"][1] == 25
    return {
        "status": "ok",
        "paper": fw["paper"],
        "psnr_trajectory100": t1["psnr"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
