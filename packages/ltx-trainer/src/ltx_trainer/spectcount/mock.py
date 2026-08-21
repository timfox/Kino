"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spectcount.config import SpectCountConfig
from ltx_trainer.spectcount.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_benchmarks,
    table3_ablation,
)


def evaluation_smoke(cfg: SpectCountConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SpectCountConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.06907"
    assert fw["signal_generation"]["fully_synthetic"] is True

    af3 = next(
        r for r in table1_benchmarks() if r["model"] == "Audio Flamingo 3" and r["setting"] == "SpectCount"
    )
    assert af3["mmau_mini_total"] == 78.40
    assert af3["mmar"] == 56.30

    full = next(r for r in table3_ablation() if "SpectCount" in r["setting"])
    assert full["accuracy"] == 78.4

    b = benchmarks_bundle()
    assert len(b["table1_auditory_benchmarks"]) == 4
    assert b["table2_signal_config"]["n_max"] == 10

    return {
        "status": "ok",
        "paper": fw["paper"],
        "mmau_mini_total": cfg.mmau_mini_total,
        "mmar": cfg.mmar,
    }
