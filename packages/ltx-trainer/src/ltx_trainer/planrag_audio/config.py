"""PlanRAG-Audio long-form audio RAG stub (arXiv:2605.20414)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanRagAudioConfig:
    paper_arxiv: str = "arXiv:2605.20414"
    title: str = "PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding"
    paper_url: str = "https://arxiv.org/abs/2605.20414"

    # Appendix F — temporal fusion tolerance (seconds)
    fusion_tau_seconds: float = 2.5

    # Evaluation audio lengths (minutes) from §4.1
    eval_durations_min: tuple[int, ...] = (10, 30, 60, 300, 540)

    # Table 4 headline — MCQA 60 min
    gemini_full_tokens_k: float = 115.2
    gemini_planrag_tokens_k: float = 0.9
    qwen_planrag_tokens_k: float = 1.2
