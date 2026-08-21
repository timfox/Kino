"""Audio reasoning survey taxonomy smoke (arXiv:2605.21008)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.audio_reasoning_survey.config import AudioReasoningSurveyConfig
from ltx_trainer.audio_reasoning_survey.formulation import joint_factorization_exists
from ltx_trainer.audio_reasoning_survey.taxonomy import four_paradigms


def evaluation_smoke(cfg: AudioReasoningSurveyConfig | None = None) -> dict[str, Any]:
    c = cfg or AudioReasoningSurveyConfig()
    paradigms = four_paradigms()
    return {
        "paper": c.paper_arxiv,
        "mmau_k_tasks": f"{c.mmau_size_k}K/{c.mmau_tasks}",
        "paradigm_count": len(paradigms),
        "reasoning_augmented": joint_factorization_exists(has_r=True),
    }
