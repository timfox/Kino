"""KIT IWSLT 2026 Instruction Following (arXiv:2606.04730)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IWSLT26IFConfig:
    paper_arxiv: str = "arXiv:2606.04730"
    title: str = (
        "Multilingual Long-Form Speech Instruction Following: KIT's Submission to IWSLT 2026"
    )
    dataset_hub: str = "YapayNet/iwslt2026-if-augmented"
    tracks: tuple[str, ...] = ("long", "short")
    languages: tuple[str, ...] = ("en", "de", "it", "zh")
    tasks: tuple[str, ...] = ("ASR", "ST", "SQA", "SSUM", "ACHAP", "MC", "Instruct")

    # Training (§4.3)
    total_samples: int = 1_048_158
    interleave_temperature: float = 2.0
    lora_rank: int = 32
    lr: float = 1e-4
    batch_size: int = 4
    max_tokens: int = 28_000
    train_steps: int = 60_000

    primary_model: str = "Qwen/Qwen2.5-Omni-7B"
    contrastive_asr: str = "nvidia/parakeet-tdt-0.6b-v2"
    contrastive_llm: str = "Qwen/Qwen2.5-7B-Instruct"

    rerank_candidates: int = 17
    rerank_en_zh: str = "Likelihood+MBR"

    # Smoke
    demo_audio_len: int = 64
    demo_text_len: int = 32
