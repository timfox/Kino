"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.detectzoo.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.04205"
    assert fw["n_detectors"] == 61
    assert fw["n_datasets"] == 22
    dz = next(r for r in demo["toolkit_table1"] if r["name"] == "DetectZoo")
    assert dz["text"] and dz["image"] and dz["audio"]
    assert demo["detector_counts"]["text"] == 36
    assert sum(demo["detector_counts"].values()) == 61
    return {
        "status": "ok",
        "paper": fw["paper"],
        "github": fw["github"],
        "n_detectors": fw["n_detectors"],
        "n_datasets": fw["n_datasets"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
