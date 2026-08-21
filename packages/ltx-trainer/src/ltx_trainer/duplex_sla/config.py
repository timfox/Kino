"""DuplexSLA full-duplex speech–language–action stub (arXiv:2605.20755)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DuplexSlaConfig:
    paper_arxiv: str = "arXiv:2605.20755"
    title: str = "DuplexSLA: A Full-Duplex Spoken Language Model with Synchronized Speech, Language, and Action"
    repo: str = "https://github.com/hyzhang24/DuplexSLA"

    # §2 — chunk clock and budgets
    chunk_ms: int = 160
    user_features_per_chunk: int = 2
    user_feature_stride_ms: int = 80
    assistant_audio_tokens_per_chunk: int = 4
    assistant_audio_stride_ms: int = 40
    action_tokens_max_per_chunk: int = 10

    backbone: str = "7B speech-LM (init Step-Audio 2 mini)"

    # DuplexSLA-Bench (§5, Table 10)
    bench_total_cases: int = 2100
    bench_turn_taking_cases: int = 1200
    bench_tool_cases: int = 900

    # Table 5 — avg delay headline (DuplexSLA vs cascade)
    tool_duplex_avg_delay_s: float = 0.64
    tool_cascade_avg_delay_s: float = 2.77
