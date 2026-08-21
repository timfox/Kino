"""VOICEGIRAFFE evaluation smoke (arXiv:2605.27976)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voicegiraffe.config import VoiceGiraffeConfig
from ltx_trainer.voicegiraffe.pipeline import pipeline_demo


def evaluation_smoke(cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    c = cfg or VoiceGiraffeConfig()
    demo = pipeline_demo(c, seed=42)
    ev = demo["eval"]
    return {
        "paper": c.paper_arxiv,
        "n_qa_total": c.n_qa_total,
        "human_overall_pct": c.human_overall_pct,
        "best_e2e_overall_pct": c.best_e2e_overall_pct,
        "only_e2e_beats_human": ev["only_e2e_beats_human"],
        "opensource_lrm_gain": ev["opensource_lrm_gain"],
        "e2e_beats_cascade": ev["e2e_beats_cascade"],
        "n_recordings_catalog": demo["hub"]["n_recordings_catalog"],
        "n_qa_full_pool": demo["hub"]["n_qa_full_pool"],
        "pool_eval_computed": demo["eval"]["pool_eval_computed"],
    }
