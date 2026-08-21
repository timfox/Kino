"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.audio_interaction.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.05121"
    assert fw["streamaudio_2m"]["items"] == 2_600_000
    assert fw["streamaudio_2m"]["categories"] == 7
    mmau = next(r for r in demo["table1_mmau"] if "Audio-Interaction" in r["model"])
    assert mmau["audio_instruction_avg"] == 58.15
    pro = next(r for r in demo["table4_proactive"] if "Audio-Interaction" in r["model"])
    assert pro["single_avg"] == 61.2
    return {
        "status": "ok",
        "paper": fw["paper"],
        "github": fw["github"],
        "dataset_hub": fw["dataset_hub"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
