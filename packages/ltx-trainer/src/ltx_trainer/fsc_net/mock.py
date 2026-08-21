"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fsc_net.config import FscNetConfig
from ltx_trainer.fsc_net.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_vctk,
    table2_ears,
    table3_ablation,
)


def evaluation_smoke(cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.06962"
    assert fw["headline"]["pesq_4k"] == 2.8092

    fsc_4k = next(r for r in table1_vctk() if r["model"] == "FSC-Net" and r["scenario"] == "4_khz_to_48_khz")
    assert fsc_4k["lsd"] == 0.8771
    assert fsc_4k["pesq"] == 2.8092

    fsc_16k = next(r for r in table1_vctk() if r["model"] == "FSC-Net" and r["scenario"] == "16_khz_to_48_khz")
    assert fsc_16k["pesq"] == 4.5279

    ears = next(r for r in table2_ears() if r["model"] == "FSC-Net")
    assert ears["pesq"] == 4.2988

    ablation = table3_ablation()
    assert len(ablation) == 3
    assert ablation[-1]["pesq"] == 2.8092

    b = benchmarks_bundle()
    assert len(b["table1_vctk"]) == 12

    return {
        "status": "ok",
        "paper": fw["paper"],
        "lsd_4k": cfg.lsd_4k,
        "pesq_ears": cfg.pesq_ears,
        "params_m": cfg.params_m,
    }
