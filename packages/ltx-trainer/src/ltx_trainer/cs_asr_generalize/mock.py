"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cs_asr_generalize.config import CsAsrGeneralizeConfig
from ltx_trainer.cs_asr_generalize.metrics import mer
from ltx_trainer.cs_asr_generalize.pipeline import benchmarks_bundle, evaluation_demo, table1_mer_results


def evaluation_smoke(cfg: CsAsrGeneralizeConfig | None = None) -> dict[str, Any]:
    c = cfg or CsAsrGeneralizeConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    baseline = next(r for r in table1_mer_results(c) if "WHISPER" in r["method"])
    ties3 = next(r for r in table1_mer_results(c) if "TIES (KO-EN + JA-EN + DE-EN)" in r["method"])
    fishr = next(r for r in table1_mer_results(c) if "Fishr" in r["method"])

    assert baseline["unseen_avg"] == 0.41
    assert ties3["seen_avg"] == 0.14
    assert ties3["unseen_avg"] == 0.34
    assert fishr["unseen_avg"] == 0.33
    assert ties3["unseen_avg"] < baseline["unseen_avg"]
    assert fishr["unseen_avg"] < baseline["unseen_avg"]
    assert c.ko_ja_eval_utterances == 450
    assert c.ko_de_eval_utterances == 387
    assert mer(["a", "b"], ["a", "c"]) == 0.5
    assert demo["improvement_vs_baseline"] == 0.07
    assert len(benchmarks_bundle(c)["table1_mer"]) == 4

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "baseline_unseen_avg_mer": c.baseline_unseen_avg,
        "ties3_unseen_avg_mer": c.ties3_unseen_avg,
        "fishr_unseen_avg_mer": c.fishr_unseen_avg,
        "ko_ja_eval_utterances": c.ko_ja_eval_utterances,
        "ko_de_eval_utterances": c.ko_de_eval_utterances,
    }
