"""Zero-shot PD detection from speech (arXiv:2605.24806)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZsPdConfig:
    paper_arxiv: str = "arXiv:2605.24806"
    sample_rate_hz: int = 16000
    segment_seconds: float = 10.0
    num_handcrafted_features: int = 71
    llm_model: str = "meta-llama/Meta-Llama-3-8B"
    lalm_models: tuple[str, ...] = (
        "Qwen/Qwen2-Audio-7B-Instruct",
        "Pengi",
        "Audio-Reasoner",
    )
    decode_temperature: float = 0.0
    random_seed: int = 0
    bootstrap_replicates: int = 10000
