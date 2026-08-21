"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.edr_zs_ser.config import EdrZsSerConfig
from ltx_trainer.edr_zs_ser.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_results,
)


def evaluation_smoke(cfg: EdrZsSerConfig | None = None) -> dict[str, Any]:
    c = cfg or EdrZsSerConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    avg = next(r for r in table2_results(c) if r.get("average"))
    assert avg["prop_uar"] == c.proposed_avg_uar
    assert avg["prop_f1"] == c.proposed_avg_f1
    assert avg["prop_uar"] > avg["b2_uar"]
    assert avg["prop_uar"] > avg["b1_uar"]

    en_de = next(r for r in table2_results(c) if r["task"] == "EN→DE")
    assert en_de["prop_uar"] == c.en_de_proposed_uar

    assert c.proposed_wo_supclr_avg_uar < c.proposed_avg_uar
    assert c.proposed_wo_spkadv_avg_uar < c.proposed_avg_uar

    assert demo["proposed_beats_baseline2_avg"]
    assert len(benchmarks_bundle()["table1_crosslingual_settings"]) == 9

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "proposed_avg_uar": c.proposed_avg_uar,
        "proposed_avg_f1": c.proposed_avg_f1,
        "en_de_proposed_uar": c.en_de_proposed_uar,
        "uar_gain_vs_baseline2": round(c.proposed_avg_uar - c.baseline2_avg_uar, 2),
    }
