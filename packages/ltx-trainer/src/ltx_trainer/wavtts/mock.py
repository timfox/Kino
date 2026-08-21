"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.wavtts.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.03455"
    wav = next(r for r in demo["table1"] if r["model"] == "WavTTS")
    assert wav["en_WER"] == 1.50
    assert wav["en_UTMOS"] == 3.92
    assert demo["headline"]["seed_en_WER"] == 1.50
    return {
        "status": "ok",
        "paper": fw["paper"],
        "github": fw["github"],
        "en_WER": wav["en_WER"],
        "en_UTMOS": wav["en_UTMOS"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
