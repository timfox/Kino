"""MuKV multi-grained KV cache compression for streaming VideoQA (arXiv:2605.22269)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MuKVConfig:
    """Stub defaults aligned with Sec. 4.1 configuration excerpt."""

    paper_arxiv: str = "arXiv:2605.22269"
    model_name: str = "LLaVA-OV"

    # Streaming setup (paper excerpt)
    fps: float = 0.5
    frames_per_segment: int = 4  # 4 continuous frames (8 seconds at 0.5 FPS)
    patches_per_frame_P: int = 196
    superpatch_factor_S: int = 4  # frame divided into S=4 super-patches

    # DCP fusion weights per granularity (patch/frame/segment)
    alpha_patch: float = 0.5
    alpha_frame: float = 0.7
    alpha_segment: float = 0.8

    # Retention ratios ρ per granularity (patch/frame/segment)
    rho_patch: float = 0.1
    rho_frame: float = 0.1
    rho_segment: float = 0.8

    # Retrieval counts (blocks) per granularity (patch/frame/segment)
    kg_patch: int = 20
    kg_frame: int = 32
    kg_segment: int = 12

    # Semi-hierarchical rerank λ_g per granularity (patch/frame/segment)
    lambda_g_patch: float = 0.3
    lambda_g_frame: float = 0.3
    lambda_g_segment: float = 0.0

    # Paper token accounting excerpt: per 10 minutes at 0.5 FPS (Table 1 style)
    rekv_mem_tokens_10min: int = 59_000
    mukv_inf_tokens: int = 8_300
    rekv_inf_tokens: int = 12_500

