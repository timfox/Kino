"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sygyt_copy.config import SygytCopyConfig
from ltx_trainer.sygyt_copy.metrics import lsd_reduction_pct
from ltx_trainer.sygyt_copy.pipeline import benchmarks_bundle, pipeline_demo, table2_copy_synthesis


def evaluation_smoke(cfg: SygytCopyConfig | None = None) -> dict[str, Any]:
    c = cfg or SygytCopyConfig()
    demo = pipeline_demo(seed=0, cfg=c)

    assert c.segments == 20
    assert c.formant_peak_error_bspline_hz == 28.0
    assert demo["hfa_reduction_matches_paper"] is True
    assert demo["berg_reduction_matches_paper"] is True
    assert demo["sublingual_dominates_ablation"] is True
    assert demo["bspline_beats_ddsp_hfa"] is True

    hfa_red = lsd_reduction_pct(c.hfa_artic_lsd, c.hfa_bspline_lsd)
    assert 29.0 <= hfa_red <= 31.0

    rows = table2_copy_synthesis(c)
    assert len(rows) == 6
    assert all(r["method"] != "B-spline" or r["lsd_db"] < 10.0 for r in rows if r["dataset"] == "HFA")

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "framework": c.framework,
        "venue": c.venue,
        "lsd_reduction_hfa_pct": round(hfa_red, 1),
        "formant_peak_error_hz": c.formant_peak_error_bspline_hz,
        "methods": len(benchmarks_bundle(c)["table4"]),
    }
