"""Framework card, paper tables, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mygardenbird.config import MygardenbirdConfig
from ltx_trainer.mygardenbird.snr import snr_demo
from ltx_trainer.mygardenbird.species import species_demo, species_table, verify_balance
from ltx_trainer.mygardenbird.split import split_demo
from ltx_trainer.mygardenbird.spectrogram import spectrogram_demo


def framework_card(cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "zenodo": c.zenodo_doi,
        "github": c.github,
        "license": c.license,
        "source": c.source_archive,
        "components": [
            "xc_metadata_fetch_and_download",
            "spectrogram_segmentation_gui",
            "manual_qc_and_snr_metadata",
            "birdnet_v24_label_check",
            "mip_source_level_split",
            "mel_cnn_edge_baseline",
        ],
        "pipeline_stages": list(c.pipeline_stages),
        "headline": headline_results(c),
    }


def table2_species_composition() -> list[dict[str, Any]]:
    """Table 2 — 12 species × 600 clips @ 16 kHz."""
    return species_table()


def table3_mip_splits() -> list[dict[str, Any]]:
    """Table 3 — 80:10:10 per-class clip and source counts (totals row)."""
    rows = [
        {"class": s["common"], "train": 480, "val": 60, "test": 60, "total": 600}
        for s in species_table()
    ]
    rows.append(
        {
            "class": "Total",
            "train": 5760,
            "val": 720,
            "test": 720,
            "total": 7200,
        }
    )
    return rows


def table6_birdnet_validation() -> list[dict[str, Any]]:
    """Table 6 — BirdNET v2.4 F1 @ 16 kHz (selected + macro)."""
    return [
        {"species": "Asian Koel", "f1_16k": 0.983, "auc_16k": 0.981},
        {"species": "Common Iora", "f1_16k": 0.973, "auc_16k": 0.974},
        {"species": "Large-tailed Nightjar", "f1_16k": 0.989, "auc_16k": 1.000},
        {"species": "Yellow-vented Bulbul", "f1_16k": 0.974, "auc_16k": 0.998},
        {"species": "Macro avg", "f1_16k": 0.979, "auc_16k": 0.991},
    ]


def table7_cnn_accuracy() -> list[dict[str, Any]]:
    """Table 7 — CNN + BirdNET accuracy (%)."""
    return [
        {"model": "MobileNetV3-Small", "acc_16k": 92.41, "acc_44k": 90.70, "std_16k": 0.64},
        {"model": "EfficientNet-B0", "acc_16k": 96.39, "acc_44k": 94.24, "std_16k": 0.69},
        {"model": "ResNet-50", "acc_16k": 94.63, "acc_44k": 93.09, "std_16k": 0.07},
        {"model": "BirdNET v2.4 (no fine-tuning)", "acc_16k": 97.94, "acc_44k": 98.06, "std_16k": None},
    ]


def table8_augmentation_16k() -> list[dict[str, Any]]:
    """Table 8 — augmentation ablations @ 16 kHz test."""
    return [
        {"model": "MobileNetV3-Small", "none": 89.40, "specaugment": 92.27, "mixup": 92.41},
        {"model": "EfficientNet-B0", "none": 94.91, "specaugment": 94.91, "mixup": 96.39},
        {"model": "ResNet-50", "none": 93.06, "specaugment": 94.07, "mixup": 94.63},
    ]


def metadata_schema() -> dict[str, Any]:
    """CSV relational schema (§3)."""
    return {
        "recordings_csv": [
            "source_id",
            "species_common",
            "species_scientific",
            "quality_grade",
            "cc_license",
            "type_label",
            "latitude",
            "longitude",
            "country",
        ],
        "clips_csv": [
            "file_id",
            "source_id",
            "onset_ms",
            "sampling_rate",
            "snr_db",
            "rms_db",
            "peak_amplitude",
            "is_clipped",
        ],
        "splits_csv": ["file_id", "split"],
    }


def curation_protocol(cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    return {
        "clip_duration_s": c.clip_duration_s,
        "clips_per_species": c.clips_per_species,
        "max_segments_per_source": c.max_segments_per_source,
        "geo_lon_bounds": (c.geo_lon_min, c.geo_lon_max),
        "split_ratios": (c.split_train, c.split_val, c.split_test),
        "solver": "CBC (COIN-OR MIP)",
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table2_species_composition": table2_species_composition(),
        "table3_mip_splits": table3_mip_splits(),
        "table6_birdnet_validation": table6_birdnet_validation(),
        "table7_cnn_accuracy": table7_cnn_accuracy(),
        "table8_augmentation_16k": table8_augmentation_16k(),
        "metadata_schema": metadata_schema(),
        "curation_protocol": curation_protocol(),
    }


def headline_results(cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    return {
        "clips_16k": c.clips_16k,
        "source_recordings": c.source_recordings,
        "snr_mean_db": c.snr_mean_db,
        "birdnet_acc_16k": c.birdnet_acc_16k,
        "efficientnet_acc_16k": c.efficientnet_acc_16k,
        "train_clips": c.train_clips,
    }


def evaluation_demo(*, seed: int = 0, cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    return {
        "framework": framework_card(c),
        "species": species_demo(),
        "balance": verify_balance(c),
        "snr": snr_demo(seed=seed, cfg=c),
        "split": split_demo(c),
        "spectrogram": spectrogram_demo(c),
        "headline": headline_results(c),
    }
