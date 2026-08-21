"""Visual counting bottleneck configuration (arXiv:2605.30170)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLMCountingConfig:
    paper_arxiv: str = "arXiv:2605.30170"
    # Synthetic toy VLM curriculum (§ 2.1)
    visual_train_max: int = 49
    text_pretrain_max: int = 99
    eval_max: int = 120
    id_max: int = 49
    ve_min: int = 50
    ve_max: int = 99
    fe_min: int = 100
    fe_max: int = 120
    board_size: int = 19
    stone_pixels: int = 14
    distractor_delta: int = 30
    # Real VLM validation (§ 2.2)
    qwen_board_size: int = 6
    qwen_stone_pixels: int = 32
    qwen_max_objects: int = 20
    foundation_model: str = "Qwen3-VL-32B-Instruct"
    embed_dim: int = 32
    probe_dim: int = 32
