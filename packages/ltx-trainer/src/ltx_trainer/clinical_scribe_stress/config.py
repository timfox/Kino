"""Paired acoustic stress test for clinical scribes — Jiang et al., arXiv:2606.05909."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClinicalScribeStressConfig:
    paper_arxiv: str = "arXiv:2606.05909"
    title: str = (
        "Beyond WER: A Paired Acoustic Stress Test for Ambient Clinical Scribes"
    )
    framework: str = "ClinicalScribeStress"

    # Corpus (§3.1)
    n_encounters: int = 272
    snr_levels_db: tuple[int, ...] = (15, 10, 5)
    asr_model: str = "whisper-large-v3"
    llm_model: str = "qwen3-235b-a22b-instruct-2507"

    # Table 2 — clean-ASR reference
    clean_wer: float = 16.54
    clean_ins_rate: float = 2.99
    clean_neg_err: float = 19.12
    clean_unsafe: float = 13.60
    clean_mean_score: float = 4.62

    # Table 2 — stationary ambient 15 dB (headline disconnect)
    ambient_15db_wer: float = 17.25
    ambient_15db_unsafe: float = 27.21
    ambient_15db_neg_err: float = 20.22
    ambient_15db_err_prop: float = 272.83

    # Table 2 — semantic MUSAN 5 dB (severe)
    semantic_5db_wer: float = 54.68
    semantic_5db_unsafe: float = 91.54

    # Table 2 — mitigation at 5 dB
    mitig_semantic_5db_unsafe: float = 70.96
    mitig_ambient_5db_unsafe: float = 33.46
    ambient_5db_unsafe: float = 40.44

    @property
    def ambient_15db_wer_delta(self) -> float:
        return round(self.ambient_15db_wer - self.clean_wer, 2)

    @property
    def ambient_15db_unsafe_ratio(self) -> float:
        return round(self.ambient_15db_unsafe / self.clean_unsafe, 2)
