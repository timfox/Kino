"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.eeg_music.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.04040"
    assert demo["headline"]["clap"] == 0.683
    assert demo["headline"]["id_50way"] == 0.487
    ours = next(r for r in demo["table1"] if r["method"] == "Ours")
    assert ours["id_50way"] == 0.487
    full = demo["table2"][0]
    assert full["id_50way"] == 0.487
    return {
        "status": "ok",
        "paper": fw["paper"],
        "github": fw["github"],
        "clap": demo["headline"]["clap"],
        "id_50way": demo["headline"]["id_50way"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
