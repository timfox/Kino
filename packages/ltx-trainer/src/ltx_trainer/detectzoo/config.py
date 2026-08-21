"""DetectZoo unified AI-generated content detection toolkit — arXiv:2606.04205."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectZooConfig:
    paper_arxiv: str = "arXiv:2606.04205"
    title: str = "DetectZoo: A Unified Toolkit for AI-Generated Content Detection"
    github: str = "https://github.com/sadjadeb/DetectZoo"
    pypi: str = "detectzoo"

    n_detectors: int = 61
    n_text_detectors: int = 36
    n_image_detectors: int = 15
    n_audio_detectors: int = 10
    n_datasets: int = 22

    default_threshold: float = 0.5
    text_max_length: int = 512
    audio_sample_rate: int = 16_000

    # Table 1 — toolkit comparison (DetectZoo is only row with all three modalities)
    unified_api: bool = True
    auto_download: bool = True

    # Reproduction anchors (Appendix / main text)
    asvspoof_res_tssdnet_eer: float = 0.012  # 1.20%
    antideepfake_hubert_eer_asvspoof: float = 0.0  # 0.00%
    fast_detectgpt_wp_auroc: float = 0.9932

    demo_batch: int = 4
