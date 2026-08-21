"""StepAudio 2.5 unified audio-language foundation (arXiv:2605.23463)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stepaudio25Config:
    paper_arxiv: str = "arXiv:2605.23463"
    family: str = "StepAudio 2.5"
    backbone: str = "frozen audio encoder + adaptor + MoE text LLM decoder"
    pretrain_tokens: str = "2.2T (text+audio multimodal curriculum)"
    mtp_branches: int = 5
    mtp_decay_alpha: float = 0.9
    asr_context_tokens: int = 32_768
    asr_rtf: float = 0.0053
    asr_zh_avg_cer: float = 2.97
    asr_en_avg_wer: float = 3.68
    asr_long_avg_er: float = 3.70
    tts_arena_win_rate: float = 0.676
    realtime_human_margin: float = 10.0
    realtime_spqa_margin: float = 16.6
