"""Parametric Memory Law + MemFT (arXiv:2605.30260)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParametricMemoryConfig:
    paper_arxiv: str = "arXiv:2605.30260"
    authors: str = "Zhejiang University / Alibaba Group"
    code_url: str = "https://github.com/zjunlp/ParametricMemoryLaw"
    models: tuple[str, ...] = ("Qwen3-8B-IT", "Llama3.1-8B-IT")
    l_crit: float = 0.6931471805599453  # ln(2), Eq. 7
    p_threshold: float = 0.5
    # Fitted law defaults (Qwen3-8B long-context combined, Table 1 style)
    law_C: float = 0.42
    law_alpha: float = 0.55
    law_beta: float = 0.48
    law_b: float = 0.02
    saturation_loss: float = 0.69
    memft_variants: tuple[str, ...] = ("SFT", "MemFT-OT", "MemFT-SW")
