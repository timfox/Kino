"""Bernini latent semantic planning for video diffusion (Bytedance, arXiv:2605.22344)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BerniniConfig:
    """Stub defaults aligned with Sec. 4.2 / 6.1 narrative."""

    paper_arxiv: str = "arXiv:2605.22344"
    project_url: str = "https://bernini-ai.github.io"
    planner_backbone: str = "Qwen2.5-VL-7B"
    renderer_backbone: str = "Wan2.2-A14B"
    planning_steps_K: int = 25
    vit_decoder_flow_steps: int = 5
    dit_steps_t2v: int = 60
    dit_steps_edit: int = 40
    flow_shift_dit: float = 5.0
    lambda_text: float = 0.2
    lambda_visual: float = 1.0
    lambda_dit: float = 1.0
    bernini_bench_cases: int = 300
    bernini_bench_categories: int = 22
    video_pair_pretrain_M: int = 20
    image_pair_pretrain_M: int = 30
