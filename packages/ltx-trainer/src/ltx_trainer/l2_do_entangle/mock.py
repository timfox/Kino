"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.l2_do_entangle.config import L2DoEntangleConfig
from ltx_trainer.l2_do_entangle.losses import cer_gap
from ltx_trainer.l2_do_entangle.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_cer_results,
)


def evaluation_smoke(cfg: L2DoEntangleConfig | None = None) -> dict[str, Any]:
    c = cfg or L2DoEntangleConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    do_row = next(r for r in table2_cer_results(c) if r.get("dual_output"))
    assert do_row["ko_meaning"] == c.ko_do_meaning_cer
    assert do_row["en_surface"] == c.en_do_surface_cer

    assert cer_gap(c.en_do_surface_cer, c.en_so_surface_cer) > 0
    assert cer_gap(c.en_do_meaning_cer, c.en_so_meaning_cer) < 0
    assert cer_gap(c.ko_do_meaning_cer, c.ko_so_meaning_cer) < 0

    assert c.en_encoder_sso_mso_layer11 < c.ko_encoder_sso_mso_layer11
    assert c.en_decoder_mso_sdo_cross_layer7 > c.en_decoder_mso_mdo_layer7

    assert demo["en_surface_degrades_under_mtl"]
    assert demo["en_meaning_improves_under_mtl"]
    assert demo["en_cross_task_inversion_layer7"]
    assert len(benchmarks_bundle()["table1_dataset_stats"]) == 2

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "ko_do_meaning_cer": c.ko_do_meaning_cer,
        "en_do_surface_cer": c.en_do_surface_cer,
        "en_surface_gap": cer_gap(c.en_do_surface_cer, c.en_so_surface_cer),
        "en_surface_gap_ed_gt10": c.en_surface_gap_ed_gt10,
    }
