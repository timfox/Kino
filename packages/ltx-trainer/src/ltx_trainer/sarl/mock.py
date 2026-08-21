"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sarl.config import SarlConfig
from ltx_trainer.sarl.pipeline import benchmarks_bundle, evaluation_demo, fig2_aggregates
from ltx_trainer.sarl.probing import baseline_normalize
from ltx_trainer.sarl.tasks import ALL_TASKS, ENCODERS


def evaluation_smoke(cfg: SarlConfig | None = None) -> dict[str, Any]:
    c = cfg or SarlConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    assert c.num_encoders == 13
    assert len(ALL_TASKS) == 7
    assert len(ENCODERS) == 13
    assert demo["source_more_sensitive"] is True
    assert demo["gram_f_source_gt_room"] is True
    assert demo["all_models_source_gt_room"] is True
    assert demo["foa_beats_mono"] is True
    assert c.gram_f_localization > c.gram_f_room
    assert c.seld_s_localization > c.seld_s_room

    phi = baseline_normalize(0.75, 0.0)
    assert abs(phi - 0.75) < 1e-6
    assert len(benchmarks_bundle(c)["table1_encoders"]) == 13
    assert len(fig2_aggregates(c)) >= 4

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "encoders": c.num_encoders,
        "probe_tasks": len(ALL_TASKS),
        "gram_f_localization": c.gram_f_localization,
        "gram_f_room": c.gram_f_room,
    }
