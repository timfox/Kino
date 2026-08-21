"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.audioim.config import AudioImConfig
from ltx_trainer.audioim.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_v2a_performance,
    table2_style_similarity,
)


def evaluation_smoke(cfg: AudioImConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioImConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.07182"
    assert fw["headline"]["ss_mos"] == 4.06

    ours_t1 = next(r for r in table1_v2a_performance() if "ours" in r["method"])
    assert ours_t1["kl_panns"] == 1.65
    assert ours_t1["ib_score"] == 31.98
    assert ours_t1["desync"] == 0.49

    ours_t2 = next(r for r in table2_style_similarity() if "ours" in r["method"])
    assert ours_t2["ss_mos"] == 4.06
    assert ours_t2["kl_passt"] == 1.63

    assert len(table1_v2a_performance()) == 4
    b = benchmarks_bundle()
    assert len(b["table2_style_similarity"]) == 4

    return {
        "status": "ok",
        "paper": fw["paper"],
        "kl_panns": cfg.kl_panns,
        "ss_mos": cfg.ss_mos,
    }
