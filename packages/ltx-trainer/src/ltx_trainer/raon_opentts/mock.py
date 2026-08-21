"""Raon-OpenTTS combined-rank filtering smoke (arXiv:2605.20830)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.raon_opentts.config import RaonOpenTtsConfig
from ltx_trainer.raon_opentts.filtering import keep_top_percentile_by_combined_rank


def evaluation_smoke(cfg: RaonOpenTtsConfig | None = None) -> dict[str, Any]:
    c = cfg or RaonOpenTtsConfig()
    rng = np.random.default_rng(0)
    n = 200
    dnsmos = rng.uniform(2.0, 4.5, n)
    speech_ratio = rng.uniform(0.6, 0.98, n)
    wer = rng.uniform(0.05, 0.45, n)
    mask = keep_top_percentile_by_combined_rank(dnsmos, speech_ratio, wer, remove_bottom_pct=15.0)
    return {
        "paper": c.paper_arxiv,
        "kept_fraction": round(float(mask.mean()), 3),
        "seed_eval_1b_wer_pct": c.seed_eval_raon_1b_wer_pct,
        "pool_hours_k": c.pool_hours_k,
    }
