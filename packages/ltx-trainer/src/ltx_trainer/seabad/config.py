"""SEABAD — Southeast Asian Bird Activity Detection dataset stub (arXiv:2605.20853)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeabadConfig:
    paper_id: str = "2605.20853"
    paper_url: str = "https://arxiv.org/abs/2605.20853"
    title: str = "SEABAD: A Tropical Bird Activity Detection Dataset for Passive Acoustic Monitoring"

    zenodo_url: str = "https://zenodo.org/records/18290494"
    curation_repo_url: str = "https://github.com/mun3im/seabad"
    validation_repo_url: str = "https://github.com/mun3im/mybad/tree/main/validation"

    # Clip format (§3, edge deployment)
    clip_duration_s: float = 3.0
    sample_rate_hz: int = 16000
    channels: int = 1
    bit_depth: int = 16
    audio_format: str = "WAV PCM"

    # Dataset scale (Table 3)
    total_clips: int = 50_000
    positive_clips: int = 25_000
    negative_clips: int = 25_000
    unique_species: int = 1677
    mean_clips_per_species_post: float = 14.9
    gini_pre_balancing: float = 0.601
    gini_post_balancing: float = 0.519
    quality_ab_fraction: float = 0.921

    # Train/val/test (§3.4)
    train_clips: int = 40_000
    val_clips: int = 5_000
    test_clips: int = 5_000

    # Segment extraction (§3.1.4)
    rms_threshold: float = 0.001
    min_temporal_separation_s: float = 1.5
    segment_hop_s: float = 0.1

    # Deduplication (§3.1.3)
    mel_bins: int = 128
    embedding_dim: int = 256
    duplicate_l2_threshold: float = 1e-7
    faiss_top_k_neighbors: int = 6

    # Balancing (§3.1.5)
    acoustic_clusters_per_species: int = 5
    salience_contrast_scale: float = 40.0

    # QA audit (§3.3)
    audit_clips_total: int = 1000
    audit_accuracy_pct: float = 97.8
    audit_margin_pct: float = 0.9

    # Baseline seeds (Table 5)
    random_seeds: tuple[int, ...] = (42, 100, 786)


@dataclass(frozen=True)
class SundalandCoverage:
    """Five-country Sundaland focus (§3.1.1)."""

    countries: tuple[str, ...] = ("Malaysia", "Indonesia", "Thailand", "Singapore", "Brunei")
    xeno_canto_records_queried: int = 43_108
    records_after_unknown_removed: int = 42_307
