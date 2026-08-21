"""Paper / integration card for Sphere360 HDR LTX bridge."""

from __future__ import annotations

from ltx_trainer.sphere360.benchmarks import PAPER_HUB, benchmarks_bundle


def framework_card() -> dict:
    b = benchmarks_bundle()
    return {
        "hub": PAPER_HUB,
        "task": "360° equirect video + FOA audio → HDR LogC3 LTX timelapse LoRA",
        "bridge": "gopex_datasets.sphere360.ltx_bridge.hdr_timelapse_lora_plan",
        "scale": b,
    }
