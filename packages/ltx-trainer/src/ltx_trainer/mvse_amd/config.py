"""Query-adaptive AV person retrieval — Loweimi et al., arXiv:2606.05931."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MvseAmdConfig:
    paper_arxiv: str = "arXiv:2606.05931"
    title: str = (
        "To Be Multimodal or Not to Be: Query-Adaptive Audio-Visual "
        "Person Retrieval via Active Modality Detection"
    )
    framework: str = "MVSE-AMD"
    affiliations: tuple[str, ...] = (
        "University of Cambridge",
        "Queen's University Belfast",
        "University of Surrey",
    )

    # Corpus (§2, §4)
    archive_videos: int = 12594
    archive_hours: float = 409.0
    query_videos: int = 523
    query_speakers: int = 38
    top_n: int = 10
    feature_dim: int = 48  # 4n + 8 for n=10

    # Presence types (§4)
    avp_queries: int = 425
    vop_queries: int = 72
    aop_queries: int = 26

    # Fusion weights (§3.1)
    lambda_aop: float = 1.0
    lambda_vop: float = 0.0
    lambda_avp: float = 0.5
    lambda_fixed: float = 0.5

    # Table 2 — modality detection LoSoCV (%)
    detect_base_acc: float = 82.3
    detect_cross_acc: float = 88.2
    detect_full_acc: float = 89.1

    # Table 3 — P@1 (%)
    speaker_p1: float = 82.9
    face_p1: float = 93.4
    fixed_p1: float = 90.0
    adaptive_p1: float = 94.2
    oracle_p1: float = 96.6
    adaptive_p3: float = 90.4
    oracle_p3: float = 91.8

    # Gap recovery (§5.2) — paper reports integer 64% from 4.2/6.6
    oracle_gap_recovery_pct: float = 64.0

    # Table 4 — adaptive P@1 by presence type (%)
    adaptive_avp_p1: float = 95.5
    adaptive_aop_p1: float = 80.8
    adaptive_vop_p1: float = 93.4


def compute_oracle_gap_recovery_pct(
    adaptive_p1: float,
    fixed_p1: float,
    oracle_p1: float,
) -> int:
    """Integer percent recovered toward oracle (matches paper rounding)."""
    gap = oracle_p1 - fixed_p1
    return round(100.0 * (adaptive_p1 - fixed_p1) / gap)
