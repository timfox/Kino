"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nnaudio2.config import NnAudio2Config
from ltx_trainer.nnaudio2.pipeline import benchmarks_bundle, pipeline_demo, table2_regression_status


def evaluation_smoke(cfg: NnAudio2Config | None = None) -> dict[str, Any]:
    c = cfg or NnAudio2Config()
    demo = pipeline_demo(seed=0, cfg=c)

    assert demo["istft_guard_raises_on_log"] is True
    assert demo["vqt_gamma_zero_routes_to_cqt"] is True
    assert demo["vqt_cqt_max_diff_at_gamma_zero"] < 1e-6
    assert demo["icqt_snr_meets_30db"] is True
    assert demo["torchscript_fix_count"] == 3
    assert c.icqt_landweber_iterations == 32
    assert c.icqt_contraction_rate == 0.8

    vqt_row = next(r for r in table2_regression_status(c) if "VQT" in r["test"])
    assert vqt_row["nnaudio2"] == "pass"

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "framework": c.framework,
        "github": c.github,
        "icqt_snr_db": demo["icqt_snr_db"],
        "fixes": len(benchmarks_bundle(c)["issue_fix"]),
    }
