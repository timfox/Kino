"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.figma.config import FigmaConfig
from ltx_trainer.figma.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_musicbench,
    table3_fmacaps_eval,
    table5_fgmcaps_test,
)


def evaluation_smoke(cfg: FigmaConfig | None = None) -> dict[str, Any]:
    c = cfg or FigmaConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    figma_mb = next(r for r in table2_musicbench() if r["model"] == "FIGMA")
    assert figma_mb["t2a_r1"] == c.musicbench_t2a_r1
    assert figma_mb["a2t_r1"] == c.musicbench_a2t_r1

    figma_fm = next(r for r in table3_fmacaps_eval() if r["model"] == "FIGMA")
    assert figma_fm["t2a_r1"] == c.fmacaps_t2a_r1

    figma_test = next(r for r in table5_fgmcaps_test() if r["model"] == "FIGMA")
    assert figma_test["t2a_r1"] == c.fgmcaps_test_t2a_r1

    assert demo["caption_saturation"]["saturates_after_50"]
    assert demo["loss"]["alpha"] == c.alpha_global
    assert len(benchmarks_bundle()["table1_dataset_comparison"]) == 4

    rel = (c.fmacaps_t2a_r1 - c.clamp3_fmacaps_t2a_r1) / c.clamp3_fmacaps_t2a_r1 * 100.0
    assert abs(rel - c.relative_improvement_fmacaps_pct) < 0.1

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "musicbench_t2a_r1": c.musicbench_t2a_r1,
        "fmacaps_t2a_r1": c.fmacaps_t2a_r1,
        "fgmcaps_test_t2a_r1": c.fgmcaps_test_t2a_r1,
        "relative_improvement_fmacaps_pct": c.relative_improvement_fmacaps_pct,
    }
