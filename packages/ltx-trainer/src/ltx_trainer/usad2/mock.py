"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.usad2.config import Usad2Config
from ltx_trainer.usad2.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_benchmark_averages,
    table2_ablation_small,
)


def evaluation_smoke(cfg: Usad2Config | None = None) -> dict[str, Any]:
    c = cfg or Usad2Config()
    demo = evaluation_demo(seed=0, cfg=c)

    xx = next(r for r in table1_benchmark_averages() if r["encoder"] == "USAD 2.0 XXLarge+")
    assert xx["hear_avg"] == c.xxlarge_plus_hear_avg
    assert xx["xares_track_a"] == c.xxlarge_plus_xares_track_a
    assert xx["xares_track_b"] == c.xxlarge_plus_xares_track_b

    usad2 = next(r for r in table2_ablation_small() if r["method"] == "USAD 2.0")
    assert usad2["nsynth_acc"] == c.usad2_nsynth_acc

    assert demo["distillation"]["domain_aware_changes_loss"]
    assert demo["beats_spear_hear"]
    assert demo["music_teacher_critical"] > 15.0
    assert len(benchmarks_bundle()["table1_benchmark_averages"]) == 7

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "xxlarge_plus_hear_avg": c.xxlarge_plus_hear_avg,
        "xxlarge_plus_xares_track_a": c.xxlarge_plus_xares_track_a,
        "xxlarge_plus_xares_track_b": c.xxlarge_plus_xares_track_b,
        "usad2_nsynth_acc": c.usad2_nsynth_acc,
    }
