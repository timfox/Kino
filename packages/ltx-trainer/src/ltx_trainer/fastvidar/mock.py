"""CPU smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fastvidar.benchmarks import benchmarks_bundle
from ltx_trainer.fastvidar.complexity import theoretical_speedup
from ltx_trainer.fastvidar.config import PAPER_ARXIV, FastViDARConfig
from ltx_trainer.fastvidar.datasets import datasets_card
from ltx_trainer.fastvidar.fastvidar_net import FastViDARStub
from ltx_trainer.fastvidar.paper import framework_card
from ltx_trainer.fastvidar.pipeline import (
    ablation_global_attention,
    evaluation_demo_run,
    fusion_strategy_demo,
    train_step,
)
from ltx_trainer.fastvidar.synthetic import synthetic_erp_frames


def evaluation_smoke() -> dict[str, Any]:
    cfg = FastViDARConfig(height=64, width=128, num_frames=4)
    model = FastViDARStub(cfg)
    frames = synthetic_erp_frames(cfg)
    out = model(frames)
    bundle = benchmarks_bundle()
    step = train_step(cfg)
    cx = theoretical_speedup()

    return {
        "package": "fastvidar",
        "status": "ok",
        "arxiv": PAPER_ARXIV,
        "delta_125_table3": bundle["table3_2d3ds"]["FastViDAR"]["delta_125"],
        "abs_rel_table3": bundle["table3_2d3ds"]["FastViDAR"]["abs_rel"],
        "beats_lightstereo_delta": bundle["table3_2d3ds"]["FastViDAR"]["delta_125"]
        > bundle["table3_2d3ds"]["LightStereo"]["delta_125"],
        "aha_beats_no_global_absrel": bundle["table1_aha"]["AHA (w+f+g)"]["abs_rel"]
        < bundle["table1_aha"]["No-Global (w+f)"]["abs_rel"],
        "theoretical_speedup": cx["speedup_factor"],
        "fusion_depth_shape": list(out["fusion_depth"].shape),
        "train_loss": step["loss"],
        "demo": evaluation_demo_run(cfg),
        "ablation": ablation_global_attention(cfg),
        "fusion": fusion_strategy_demo(cfg),
        "datasets": datasets_card(),
        "framework": framework_card()["components"],
    }
