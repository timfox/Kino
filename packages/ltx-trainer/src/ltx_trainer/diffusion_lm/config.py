"""Paper metadata for Diffusion LM agent integration."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_BITTER_LESSON = "2601.12979"
PAPER_DLLM_AGENT = "2602.07451"
PAPER_DLLM_SEARCHER = "2602.07035"
PAPER_DLMASR = "2605.29613"


@dataclass
class DiffusionLmPaperConfig:
    bitter_lesson_arxiv: str = PAPER_BITTER_LESSON
    dllm_agent_arxiv: str = PAPER_DLLM_AGENT
    dllm_searcher_arxiv: str = PAPER_DLLM_SEARCHER
    dlmasr_arxiv: str = PAPER_DLMASR
    default_backbone: str = "autoregressive"
    default_aux_role: str = "memory_summarize"
