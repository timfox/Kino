"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.glass.config import GlassConfig
from ltx_trainer.glass.pipeline import benchmarks_bundle, evaluation_demo, table1_individual_control


def evaluation_smoke(cfg: GlassConfig | None = None) -> dict[str, Any]:
    c = cfg or GlassConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    baseline = next(r for r in table1_individual_control(c) if "Baseline" in r["method"])
    fast = next(r for r in table1_individual_control(c) if r["method"] == "Fast LoRA (ours)")
    slow = next(r for r in table1_individual_control(c) if r["method"] == "Slow LoRA (ours)")

    assert fast["sps"] > baseline["sps"]
    assert slow["sps"] < baseline["sps"]
    assert abs(fast["wer"] - baseline["wer"]) <= 0.7
    assert abs(slow["wer"] - baseline["wer"]) <= 0.7
    assert c.interp_speed_wer_min < baseline["wer"]
    assert c.interp_pitch_wer_min < baseline["wer"]
    assert c.lora_fraction_pct == 0.22
    assert demo["grpo_group_size"] == 8
    assert len(benchmarks_bundle(c)["table1_individual"]) == 5

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "baseline_wer": c.baseline_wer,
        "fast_sps": c.fast_sps,
        "slow_sps": c.slow_sps,
        "interp_wer_min_speed": c.interp_speed_wer_min,
        "interp_wer_min_pitch": c.interp_pitch_wer_min,
        "lora_fraction_pct": c.lora_fraction_pct,
    }
