"""Config for VLM visual counting bottleneck stub (arXiv:2605.30170)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VlmCountConfig:
    paper_arxiv: str = "2605.30170"
    visual_train_max: int = 49  # synthetic toy VLM Phase 2
    text_train_max: int = 99  # Phase 1 language pretrain
    full_extrap_max: int = 120
    qwen_grid: int = 6
    qwen_visual_max: int = 20
    probe_dim: int = 32
    stages: tuple[str, ...] = ("visual_individuation", "magnitude_awareness", "symbolic_mapping")


DEFAULT_CONFIG = VlmCountConfig()
