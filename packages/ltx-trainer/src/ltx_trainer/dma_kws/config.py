"""DMA-KWS user-defined keyword spotting stub (arXiv:2605.22120).

Reference implementation notes:
- This module is a GOPEX paper stub (toy math + table excerpts).
- It does NOT include model weights, Conformer training, LoRA training loops, or datasets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DmaKwsConfig:
    paper_arxiv: str = "arXiv:2605.22120"
    title: str = "DMA-KWS: dual-stage matching, multimodal enrollment, continual adaptation"

    # Paper headline: SI-KWS LibriPhrase Hard (LPH) under LS-GS-1460 + LP-GP-1460
    headline_auc_lph: float = 97.85
    headline_eer_lph: float = 6.13

    # Parameter-efficient continual adaptation (paper text)
    lora_updated_params_total: int = 187_000
    lora_updated_params_stage1: int = 114_000
    lora_updated_params_stage2: int = 73_000

    # CTC phoneme inventory size (70 phonemes + blank)
    phoneme_vocab_size: int = 71

    # A small subset of the reported evaluation targets in the paper
    eval_sets: tuple[str, ...] = ("LibriPhrase (LPE/LPH)", "GSC", "Qcomm", "Hey-Snips", "DeepMine")

