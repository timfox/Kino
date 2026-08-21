"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clean_codec.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    t1 = {r["codec"]: r for r in demo["table1"]}
    t5 = {r["codec"]: r for r in demo["table5_tts"]}
    assert fw["paper"] == "arXiv:2606.04418"
    assert fw["codebook_size"] == 32_768
    assert t1["CleanCodec@12.5"]["SIM"] == 0.86
    assert t1["CleanCodec@12.5"]["WER"] == 2.7
    assert t5["CleanCodec"]["WER"] == 3.9
    return {
        "status": "ok",
        "paper": fw["paper"],
        "sim_12_5": t1["CleanCodec@12.5"]["SIM"],
        "wer_12_5": t1["CleanCodec@12.5"]["WER"],
        "tts_wer": t5["CleanCodec"]["WER"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
