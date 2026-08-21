"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.wat.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.02631"
    best = next(r for r in demo["table1"] if "audio scale 4" in r["model"])
    assert best["A-PSNR"] == 39.92
    assert best["V-PSNR"] == 23.93
    vid = next(r for r in demo["table5"] if r["modality"] == "Video")
    assert vid["PSNR"] == 34.45
    return {
        "status": "ok",
        "paper": fw["paper"],
        "audio_psnr": best["A-PSNR"],
        "video_psnr_dense": best["V-PSNR"],
        "masked_video_psnr": vid["PSNR"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
