"""StreamOV configuration (arXiv:2605.25621)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamOVConfig:
    """Reference hyperparameters from the StreamOV paper."""

    paper_arxiv: str = "arXiv:2605.25621"
    backbone: str = "Qwen3-Omni-30B-A3B"
    trigger_params_m: float = 18.9
    frame_budget_sovbench: int = 64
    frame_budget_other: int = 32
    short_memory_k: int = 8
    long_memory_k: int = 16
    trigger_lr: float = 3e-4
    trigger_batch_size: int = 32
    hidden_dim: int = 256
    sovbench_o_sessions: int = 172
    sovbench_o_turns: int = 1739
    sovbench_t_samples: int = 226
