"""NeurOWL configuration — Yang et al., arXiv:2607.15776."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.15776"
PAPER_TITLE = (
    "NeurOWL: An LLM-Based Neural-symbolic Framework for "
    "Incomplete OWL Ontology Reasoning"
)
PAPER_AUTHORS = (
    "Hui Yang, Jiaoyan Chen, Yiping Song, Renate Schmidt, Wen Zhang"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_ONT_REPO = "https://github.com/HuiYang1997/OnT"
BENCHMARK = "FoodOnA / SnomedA / Snomed∃ (1k pos + 1k neg each)"

STAGES = ("1", "2a", "2b", "3a", "3b")


@dataclass
class NeurOWLConfig:
    """Runtime knobs for the CPU stub."""

    embedding_backend: str = "ont"  # ont | sbert
    llm_name: str = "Qwen3.5-9B"
    top_k: int = 10
    max_missing_axioms: int = 2
    enable_stage_2a: bool = True
    enable_stage_2b: bool = True
    enable_stage_3a: bool = True
    enable_stage_3b: bool = True
    fine_tuned: bool = True
