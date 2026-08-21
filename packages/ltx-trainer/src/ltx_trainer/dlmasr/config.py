"""DLM-ASR decoding strategies (arXiv:2605.29613)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecodingStrategy(str, Enum):
    FIXED_NUMBER = "fixed_number"
    STATIC_THRESHOLD = "static_threshold"
    DYNAMIC_THRESHOLD = "dynamic_threshold"
    AUTOREGRESSIVE = "autoregressive"


@dataclass
class DlmAsrConfig:
    paper_arxiv: str = "arXiv:2605.29613"
    speech_encoder: str = "Whisper-medium.en"
    dlm_decoder: str = "LLaDA-8B-Instruct"
    baseline_system: str = "Whisper-LLaDA"
    train_corpus: str = "LibriSpeech-960h"
    eval_split: str = "test-clean"
    train_steps: int = 100_000
    block_size_default: int = 4
    # Fig. 1 / Sec. 4.1 anchors
    ar_wer_pct: float = 2.78
    static_b4_c095_wer_pct: float = 2.81
    static_b4_c095_rtf_speedup: float = 1.7
    static_parallel_wer_pct: float = 4.13
    static_parallel_speedup: float = 3.5
    # Fig. 2(d) matched RTF WER
    matched_rtf_static_wer_pct: float = 3.07
    matched_rtf_fixed_wer_pct: float = 4.47
    matched_rtf_dynamic_wer_pct: float = 3.52
    # Fig. 3 mean stopping rounds + RTF
    static_mean_rounds: float = 6.1
    dynamic_mean_rounds: float = 9.5
    fixed_k1_mean_rounds: float = 32.0
    static_rtf: float = 0.046
    dynamic_rtf: float = 0.081
    fixed_rtf: float = 0.298
    # Fig. 4 CCDF anchors (fraction >= threshold)
    asr_frac_ge_090: float = 0.937
    asr_frac_ge_095: float = 0.911
    gsm8k_frac_ge_090: float = 0.591
    gsm8k_frac_ge_095: float = 0.457
