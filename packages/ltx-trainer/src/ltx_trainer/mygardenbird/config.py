"""MyGardenBird — ML-ready Malaysian bird sound dataset (arXiv:2606.06975)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MygardenbirdConfig:
    paper_arxiv: str = "arXiv:2606.06975"
    title: str = (
        "MyGardenBird: A Machine-Learning-Ready Bird Sound Dataset for "
        "Twelve Common Malaysian Birds"
    )
    framework: str = "MyGardenBird"
    zenodo_doi: str = "https://doi.org/10.5281/zenodo.20306877"
    github: str = "https://github.com/mun3im/MyGardenBird"
    license: str = "CC BY-NC-SA 4.0"
    source_archive: str = "Xeno-canto"

    # Core release (§3)
    n_species: int = 12
    clips_per_species: int = 600
    clips_16k: int = 7200
    clips_44k: int = 6950
    source_recordings: int = 1381
    clip_duration_s: float = 3.0
    sample_rate_16k: int = 16000
    sample_rate_44k: int = 44100
    total_hours: float = 6.0

    # SNR (§4.1)
    snr_min_db: float = 0.83
    snr_max_db: float = 59.18
    snr_mean_db: float = 15.80
    snr_std_db: float = 9.10

    # Split (§2.6, Table 3)
    split_train: float = 0.8
    split_val: float = 0.1
    split_test: float = 0.1
    train_clips: int = 5760
    val_clips: int = 720
    test_clips: int = 720
    clips_per_class_split: int = 60  # per partition per species

    # BirdNET validation (§4.4, Table 6)
    birdnet_acc_16k: float = 97.94
    birdnet_acc_44k: float = 98.06
    birdnet_macro_auc_16k: float = 0.9913

    # CNN baselines Table 7 @ 16 kHz (Mixup)
    mobilenet_acc_16k: float = 92.41
    efficientnet_acc_16k: float = 96.39
    resnet50_acc_16k: float = 94.63

    # Segmentation / QC
    max_segments_per_source: int = 10
    geo_lon_min: float = 60.0
    geo_lon_max: float = 140.0

    pipeline_stages: tuple[str, ...] = (
        "Stage1_xc_fetch_metadata",
        "Stage2_download",
        "Stage3_audit",
        "Stage4_annotate_segments",
        "Stage5_extract_clips",
        "Stage6_qc",
        "Stage7_birdnet_validation",
        "Stage8_splitter_mip",
        "Stage9_train_mygardenbird_multifeature",
    )
