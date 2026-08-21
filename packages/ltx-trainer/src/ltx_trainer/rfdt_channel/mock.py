"""CPU smoke for paper stub validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rfdt_channel.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    cmp = demo["material_comparison"]
    assert fw["paper"] == "arXiv:2606.01261"
    assert cmp["dominant_path_unchanged"]
    assert cmp["path_reduction"] == 742 - 52
    assert demo["fig3"]["effective_path_count"]["All_Concrete"] == 742
    assert demo["reconstruction"]["pipeline"] == ["COLMAP", "3DGS", "SuGaR"]
    return {
        "status": "ok",
        "paper": fw["paper"],
        "scene": fw["scene"],
        "paths_multi_material": demo["fig3"]["effective_path_count"]["Multi_Material"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
