"""PagedWeight — dynamic quality-aware MoE weight quantization (arXiv:2607.16184)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.16184"
PAPER_TITLE = "PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization"
PAPER_SYSTEM = "PagedWeight"
PAPER_AUTHORS = "Yuchen Yang, Yifan Zhao, Anisha Dasgupta, Sasa Misailovic (UIUC)"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
BENCHMARK = "Qwen1.5-MoE-A2.7B / Mixtral-8×7B / Gemma-4-26B-A4B (Wikitext2, C4, GSM8K, MATH, LongBench)"

COMPONENTS = (
    "weight_page_table",
    "quality_aware_planner",
    "async_page_movement",
    "fused_mixed_precision_moe_kernel",
)

SUPPORTED_BITWIDTHS = (3, 4, 5, 6, 7, 8)
LINEAR_BLOCKS = ("gate_up", "down")


@dataclass
class PagedWeightConfig:
    """Runtime knobs for the CPU stub."""

    n_experts: int = 8
    n_layers: int = 4
    free_block_threshold: int = 4
    kv_block_bytes: int = 1024 * 1024
    residual_strength: float = 0.5
    enable_routing: bool = True
    enable_prompt_residual: bool = True
    enable_page_movement: bool = True
    enable_global_sensitivity: bool = True
