"""MDD-LSSG — Tu et al., arXiv:2606.05569."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MddLssgConfig:
    paper_arxiv: str = "arXiv:2606.05569"
    title: str = (
        "Domain-Aware Mispronunciation Detection and Diagnosis Using "
        "Language-Specific Statistical Graphs"
    )
    framework: str = "MDD-LSSG"
    venue: str = "Interspeech-style / arXiv"

    audio_encoder: str = "facebook/wav2vec2-large-xlsr-53"
    gcn_layers: int = 2
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_epochs: int = 100
    embed_dim: int = 64
    dropout: float = 0.1

    dataset: str = "L2-ARCTIC"
    num_speakers: int = 24
    l1_backgrounds: tuple[str, ...] = (
        "Arabic",
        "Hindi",
        "Korean",
        "Mandarin",
        "Spanish",
        "Vietnamese",
    )
    test_speakers: tuple[str, ...] = ("NJS", "TXHC", "TLV", "ZHAA", "YKWK", "TNI")

    # Table 1 — mispronunciation detection
    ours_detection_recall: float = 0.5779
    ours_detection_precision: float = 0.6136
    ours_detection_f1: float = 0.5952

    aux_embed_f1: float = 0.5641
    lookup_embed_f1: float = 0.5683
    mddgcn_f1: float = 0.5649
    cat_gcn_f1: float = 0.5824

    # Table 2 — mispronunciation diagnosis
    ours_frr: float = 6.02
    ours_far: float = 42.21
    ours_der: float = 20.88

    aux_embed_der: float = 20.98
    lookup_embed_der: float = 21.45
    mddgcn_der: float = 25.24
    cat_gcn_der: float = 20.84

    # Fig. 3 per-L1 F1 anchors (ours vs CAT-GCN-MDD on Spanish — largest gap)
    spanish_ours_f1: float = 0.62
    spanish_cat_gcn_f1: float = 0.55
    spanish_lookup_f1: float = 0.57
