"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.aac_audioset.config import AacAudiosetConfig
from ltx_trainer.aac_audioset.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_clotho,
    table3_keyword_ablation,
    table4_k_sensitivity,
)


def evaluation_smoke(cfg: AacAudiosetConfig | None = None) -> dict[str, Any]:
    c = cfg or AacAudiosetConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    ours = next(r for r in table1_clotho() if r["method"] == "Ours" and r["setting"] == "in-domain")
    assert ours["spider"] == c.clotho_id_spider
    assert ours["fense"] == c.clotho_id_fense

    t3 = next(
        r for r in table3_keyword_ablation() if r["decoder"] == "Ours" and r["keyword_guidance"]
    )
    assert t3["spider"] == c.ours_kw_spider

    k5 = next(r for r in table4_k_sensitivity() if r["k"] == 5)
    assert k5["spider"] == c.k5_spider

    assert demo["keyword_guidance_improves_spider"]
    assert demo["beats_llama_keywords_only"]
    assert len(benchmarks_bundle()["table2_audiocaps"]) == 6

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "clotho_spider": c.clotho_id_spider,
        "audiocaps_spider": c.audiocaps_id_spider,
        "top_k_keywords": c.top_k_keywords,
    }
