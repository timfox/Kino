"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.xvec_emo.config import XvecEmoConfig
from ltx_trainer.xvec_emo.pipeline import benchmarks_bundle, pipeline_demo, table2_en_held_out


def evaluation_smoke(cfg: XvecEmoConfig | None = None) -> dict[str, Any]:
    c = cfg or XvecEmoConfig()
    demo = pipeline_demo(seed=0, cfg=c)

    assert demo["localized_operand"].startswith("x-vector")
    assert demo["en_delta_matches_paper"] is True
    assert demo["avg4spk_identity_better"] is True
    assert demo["secs_floor_met"] is True
    assert c.delta_eecs_en_avg4spk == 0.288
    assert c.cos_neutral_angry == 0.988
    assert len(c.esd_tau_speakers) == 4

    angry = next(r for r in table2_en_held_out(c) if r["emotion"] == "angry")
    assert angry["avg4spk_eecs"] == 0.925

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "framework": c.framework,
        "backbone": c.backbone,
        "delta_eecs_en": c.delta_eecs_en_avg4spk,
        "delta_eecs_ptbr": c.delta_eecs_ptbr,
        "elimination_steps": len(benchmarks_bundle(c)["elimination"]),
    }
