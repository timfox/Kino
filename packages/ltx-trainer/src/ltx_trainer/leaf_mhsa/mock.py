"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.leaf_mhsa.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    summary = demo["table2_summary"]
    assert fw["doi"] == "10.1117/12.3061298"
    assert fw["n_traits"] == 16
    assert summary["r2"] >= 0.84
    assert summary["nrmse_pct"] <= 1.52
    return {
        "status": "ok",
        "doi": fw["doi"],
        "r2_avg": summary["r2"],
        "nrmse_pct_avg": summary["nrmse_pct"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
