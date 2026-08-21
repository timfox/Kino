"""Cause-aware SDS clarification smoke (arXiv:2605.25404)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cause_sds.clarification import k_round_clarification
from ltx_trainer.cause_sds.config import CauseSdsConfig
from ltx_trainer.cause_sds.detectors import ErrorCause
from ltx_trainer.cause_sds.tsallis import tsallis_token_confidence


def evaluation_smoke(cfg: CauseSdsConfig | None = None) -> dict[str, Any]:
    c = cfg or CauseSdsConfig()
    clar = k_round_clarification(
        "book Dunzo",
        [("Dunzo", ErrorCause.COMPREHENSION)],
        ["Dunzo delivery"],
        k_max=c.clarification_rounds,
    )
    uniform = np.ones(c.vocab_size) / c.vocab_size
    conf = tsallis_token_confidence(uniform, alpha=c.tsallis_alpha)
    return {
        "paper": c.paper_arxiv,
        "comprehension_recall_spgi2_pct": 57.96,
        "wsj_wer_after_3_rounds": 3.85,
        "clarification_rounds_run": len(clar["rounds"]),
        "tsallis_uniform_conf": round(conf, 4),
    }
