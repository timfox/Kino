"""TaC / TaC-C configuration (arXiv:2605.28713)."""

from __future__ import annotations

from dataclasses import dataclass

_THINK_OPEN = "<" + "redacted_thinking" + ">"
_THINK_CLOSE = "</" + "redacted_thinking" + ">"


@dataclass
class TaCConfig:
    """Thinking as Compression settings."""

    max_context_tokens: int = 8192
    max_thinking_tokens: int = 2048
    compression_ratios: tuple[int, ...] = (4, 8)
    budget_gamma: float = 0.1  # soft gate tolerance (Eq. 3)
    lambda_format: float = 0.05
    lambda_utility: float = 0.95
    grpo_group_size: int = 8
    # Paper headline averages (Table 1, Qwen3-8B TaC-C)
    paper_em_4x: float = 51.17
    paper_f1_4x: float = 65.12
    paper_em_8x: float = 51.79
    paper_f1_8x: float = 65.42
    thinking_open: str = _THINK_OPEN
    thinking_close: str = _THINK_CLOSE
    datasets: tuple[str, ...] = ("natural_questions", "2wikimqa", "hotpotqa", "musique")
