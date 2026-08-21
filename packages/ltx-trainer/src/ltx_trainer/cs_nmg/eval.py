"""End-to-end CS-NMG pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cs_nmg.config import CsNmgConfig
from ltx_trainer.cs_nmg.pipeline import evaluation_demo, headline_results, table2_main_results


def pipeline_demo(*, seed: int = 0, cfg: CsNmgConfig | None = None) -> dict[str, Any]:
    c = cfg or CsNmgConfig()
    demo = evaluation_demo(seed=seed, cfg=c)
    best = next(r for r in table2_main_results() if "tri-level" in r["method"])
    return {
        "demo": demo,
        "best_cmn_wer": best["cmn_wer"],
        "best_vie_pier": best["vie_pier"],
        "contrastive_improves_over_wce": best["cmn_wer"] < 16.42,
        "headline": headline_results(c),
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["contrastive_improves_over_wce"]
    assert out["best_cmn_wer"] == 14.06
    return {"status": "ok", "cmn_wer": out["best_cmn_wer"]}
