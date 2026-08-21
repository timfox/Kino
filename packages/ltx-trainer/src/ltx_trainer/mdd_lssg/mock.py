"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mdd_lssg.config import MddLssgConfig
from ltx_trainer.mdd_lssg.graph import build_statistical_graph, demo_substitution_counts_vietnamese
from ltx_trainer.mdd_lssg.pipeline import benchmarks_bundle, evaluation_demo, table1_detection


def evaluation_smoke(cfg: MddLssgConfig | None = None) -> dict[str, Any]:
    c = cfg or MddLssgConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    t1 = table1_detection(c)

    assert c.ours_detection_f1 == 0.5952
    assert c.ours_der == 20.88
    assert demo["directional_asymmetric"] is True
    assert demo["statistical_closer_confused_pair"] is True
    assert demo["beats_baselines"] is True
    assert demo["der_beats_mddgcn"] is True

    ours = next(r for r in t1 if "Ours" in r["model"])
    assert ours["f1"] == 0.5952
    assert len(benchmarks_bundle(c)["table_l1_f1"]) == 6

    g = build_statistical_graph(demo_substitution_counts_vietnamese())
    assert abs(g.outgoing_sum("z") - 1.0) < 1e-6
    assert g.edge_weight("s", "z") > g.edge_weight("z", "s")

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "detection_f1": c.ours_detection_f1,
        "diagnosis_der": c.ours_der,
        "dataset": c.dataset,
        "l1_count": len(c.l1_backgrounds),
    }
