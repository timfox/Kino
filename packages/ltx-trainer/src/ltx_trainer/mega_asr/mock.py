"""Mega-ASR DG-WGPO reward smoke (arXiv:2605.19833)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mega_asr.config import MegaASRConfig
from ltx_trainer.mega_asr.rewards import token_edit_similarity, word_wer


def evaluation_smoke(cfg: MegaASRConfig | None = None) -> dict[str, Any]:
    c = cfg or MegaASRConfig()
    sim = token_edit_similarity("hello", "helo")
    wer = word_wer("the cat sat", "the cat sit")
    return {
        "paper": c.paper_arxiv,
        "token_similarity": round(sim, 4),
        "word_wer": round(wer, 4),
    }
