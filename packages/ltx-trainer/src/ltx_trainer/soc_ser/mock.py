"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.soc_ser.config import SocSerConfig
from ltx_trainer.soc_ser.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    lem_ablation_delta,
    table1_esd_ravdess,
)


def evaluation_smoke(cfg: SocSerConfig | None = None) -> dict[str, Any]:
    c = cfg or SocSerConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    hubert = next(
        r for r in table1_esd_ravdess() if r["backbone"] == "HuBERT" and r["method"] == "SOC"
    )
    assert hubert["esd_wa"] == c.hubert_soc_esd_wa
    assert hubert["ravdess_wa"] == c.hubert_soc_ravdess_wa

    w2v_soc = next(
        r for r in table1_esd_ravdess() if r["backbone"] == "Wav2Vec2" and r["method"] == "SOC"
    )
    w2v_gap = next(
        r for r in table1_esd_ravdess() if r["backbone"] == "Wav2Vec2" and r["method"] == "GAP"
    )
    assert abs((w2v_soc["esd_wa"] - w2v_gap["esd_wa"]) - c.w2v_esd_gain_pct) < 0.01

    assert demo["soc_layer"]["vector_dim"] == demo["soc_layer"]["expected_vector_dim"]
    assert demo["soc_layer"]["lem_differs_from_no_lem"]
    assert abs(lem_ablation_delta(c)["esd_wa_drop_pct"] - 1.45) < 0.01
    assert len(benchmarks_bundle()["table1_esd_ravdess"]) == 15

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "hubert_soc_esd_wa": c.hubert_soc_esd_wa,
        "hubert_soc_ravdess_wa": c.hubert_soc_ravdess_wa,
        "w2v_esd_gain_pct": c.w2v_esd_gain_pct,
        "w2v_ravdess_gain_pct": c.w2v_ravdess_gain_pct,
    }
