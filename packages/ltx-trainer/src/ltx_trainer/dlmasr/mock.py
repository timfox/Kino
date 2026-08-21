"""DLM-ASR decoding evaluation smoke (arXiv:2605.29613)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dlmasr.config import DlmAsrConfig
from ltx_trainer.dlmasr.pipeline import pipeline_demo


def evaluation_smoke(cfg: DlmAsrConfig | None = None) -> dict[str, Any]:
    c = cfg or DlmAsrConfig()
    demo = pipeline_demo(c, seed=42)
    strat = demo["strategy_compare"]
    return {
        "paper": c.paper_arxiv,
        "baseline": c.baseline_system,
        "static_b4_wer_pct": strat["static_wer_pct"],
        "static_fewer_rounds": strat["static_fewer_rounds"],
        "static_beats_fixed_wer": strat["static_beats_fixed_wer"],
        "best_strategy": strat["best_strategy"],
        "asr_frac_ge_095": c.asr_frac_ge_095,
    }
