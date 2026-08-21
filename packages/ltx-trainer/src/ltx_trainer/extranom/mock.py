"""ExtrAnom caption metrics smoke (arXiv:2605.25806)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.extranom.config import ExtrAnomConfig
from ltx_trainer.extranom.metrics import rouge_l_recall, unigram_f1


def evaluation_smoke(cfg: ExtrAnomConfig | None = None) -> dict[str, Any]:
    c = cfg or ExtrAnomConfig()
    f1 = unigram_f1("woman walking alone at night", "woman walks alone night street")
    rouge = rouge_l_recall("woman walking alone", "woman walks alone at night")
    return {
        "paper": c.paper_arxiv,
        "caption_f1": round(f1, 3),
        "rouge_l_recall": round(rouge, 3),
    }
