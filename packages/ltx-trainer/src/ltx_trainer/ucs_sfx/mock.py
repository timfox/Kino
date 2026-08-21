"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ucs_sfx.cascade import classify_tag
from ltx_trainer.ucs_sfx.config import UcsSfxConfig
from ltx_trainer.ucs_sfx.pipeline import benchmarks_bundle, evaluation_demo, table1_conversion


def evaluation_smoke(cfg: UcsSfxConfig | None = None) -> dict[str, Any]:
    c = cfg or UcsSfxConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    t1 = table1_conversion(c)

    assert c.envsound_total == 58057
    assert c.envsound_categories == 59
    assert c.ucs_synonyms == 9972
    assert abs(t1[0]["rate"] - 1.0) < 1e-6
    assert abs(t1[1]["rate"] - 0.9849) < 1e-4
    assert abs(t1[2]["rate"] - 1.0) < 1e-6
    assert c.envsound_subcat_hier_oracle_f1 == 0.73
    assert demo["split_passes_threshold"] is True
    assert demo["easy_category"] == "ANIMALS"
    assert demo["conflict_category"] == "VOICES"
    assert len(benchmarks_bundle(c)["table1_conversion"]) == 3

    hit = classify_tag(
        "gunshot_and_gunfire",
        mapping={"gunshot and gunfire": ("GUNS", "GUNSHOT")},
    )
    assert hit.category == "GUNS"
    assert hit.stage == "predefined"

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "envsound_total": c.envsound_total,
        "envsound_categories": c.envsound_categories,
        "fsd50k_conversion_rate": c.fsd50k_classified_rate,
        "audioset_conversion_rate": c.audioset_classified_rate,
        "envsound_subcat_flat_f1": c.envsound_subcat_flat_f1,
    }
