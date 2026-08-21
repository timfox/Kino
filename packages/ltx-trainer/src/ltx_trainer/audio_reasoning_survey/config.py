"""Config for Audio Reasoning survey stub (arXiv:2605.21008)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioReasoningSurveyConfig:
    paper_arxiv: str = "arXiv:2605.21008"
    title: str = "A Survey of Audio Reasoning in Multimodal Foundation Models"

    # Headline numbers from paper tables (spot-check anchors)
    mmau_size_k: int = 10
    mmau_tasks: int = 27
    mmau_pro_size_k: float = 0.5
    mmau_pro_tasks: int = 49
    voiceagentbench_size_k: int = 6
    voiceagentbench_tasks: int = 6
