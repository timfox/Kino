"""ExtrAnom dataset configuration (Sangeeta et al., arXiv:2605.25806)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtrAnomConfig:
    """Women-centric multi-modal VAD benchmark defaults."""

    paper_arxiv: str = "arXiv:2605.25806"
    repo_url: str = "https://github.com/A24CS09005/ExtrAnom"

    total_videos: int = 1001
    normal_videos: int = 500
    anomalous_videos: int = 501
    anomaly_classes: int = 5
    textual_annotations_per_video: int = 4
    """One human ground truth + three LLM descriptions (ChatGPT, DeepSeek, Mistral)."""

    train_videos: int = 800
    train_normal: int = 400
    train_anomalous: int = 400
    test_videos: int = 201
    test_normal: int = 100
    test_anomalous: int = 101
    train_ratio: float = 0.8

    llm_annotators: tuple[str, ...] = ("ChatGPT", "DeepSeek", "Mistral")

    # Approximate shares among anomalous videos (Sec. 3.4, Fig. 1)
    pct_low_light: float = 8.0
    pct_low_resolution: float = 13.0
    pct_long_shot: float = 15.0
    pct_daylight_well_lit: float = 64.0

    benchmark_datasets: tuple[str, ...] = field(
        default_factory=lambda: ("RareAnom", "UCF-Crime", "XD-Violence", "UCA")
    )
