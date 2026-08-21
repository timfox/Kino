"""Framework card, dataset comparison tables, and benchmark excerpts (arXiv:2605.20853)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.seabad.balancing import gini_reduction_pct
from ltx_trainer.seabad.config import SeabadConfig, SundalandCoverage
from ltx_trainer.seabad.layout import LIMITATIONS
from ltx_trainer.seabad.mock import evaluation_smoke


def table1_dataset_comparison() -> list[dict[str, Any]]:
    """Table 1 — bird audio datasets for detection/classification (excerpt)."""
    return [
        {
            "dataset": "Freefield1010",
            "region": "Global",
            "task": "Detection",
            "clips": 7690,
            "species": "unlabeled",
            "clip_length": "10 s",
            "annotation": "Crowd",
        },
        {
            "dataset": "Warblr",
            "region": "UK",
            "task": "Detection",
            "clips": 10000,
            "species": "20+",
            "clip_length": "10 s",
            "annotation": "Crowd",
        },
        {
            "dataset": "BirdVox-DCASE",
            "region": "N. America",
            "task": "Detection",
            "clips": 20000,
            "species": "<10",
            "clip_length": "10 s",
            "annotation": "Expert",
        },
        {
            "dataset": "BirdSet",
            "region": "Global",
            "task": "Classification",
            "clips": "520k+",
            "species": "10k+",
            "clip_length": "variable",
            "annotation": "Mixed",
        },
        {
            "dataset": "SEABAD",
            "region": "SE Asia",
            "task": "Detection",
            "clips": 50000,
            "species": "1600+",
            "clip_length": "3 s",
            "annotation": "Mixed",
        },
    ]


def table2_negative_sources() -> list[dict[str, Any]]:
    """Table 2 — public datasets for non-bird (negative) clips."""
    return [
        {"dataset": "BirdVox-DCASE-20k", "usable": 9983, "geography": "NE USA"},
        {"dataset": "Freefield1010", "usable": 5755, "geography": "Global"},
        {"dataset": "Warblr", "usable": 1950, "geography": "UK"},
        {"dataset": "FSC-22", "usable": 1875, "geography": "European forests"},
        {"dataset": "ESC-50", "usable": 1840, "geography": "Global"},
        {"dataset": "DataSEC", "usable": 3597, "geography": "Southern Europe"},
    ]


def table3_dataset_statistics() -> dict[str, Any]:
    """Table 3 — SEABAD dataset statistics."""
    cfg = SeabadConfig()
    return {
        "total_clips": cfg.total_clips,
        "positive_clips": cfg.positive_clips,
        "negative_clips": cfg.negative_clips,
        "unique_bird_species": cfg.unique_species,
        "clip_duration_s": cfg.clip_duration_s,
        "sample_rate_hz": cfg.sample_rate_hz,
        "format": f"{cfg.audio_format} {cfg.bit_depth}-bit",
        "mean_samples_per_species": cfg.mean_clips_per_species_post,
        "gini_post_balancing": cfg.gini_post_balancing,
        "quality_rating_A_or_B_pct": round(100.0 * cfg.quality_ab_fraction, 1),
    }


def table4_geographic_distribution() -> list[dict[str, Any]]:
    """Table 4 — positive samples before/after balancing (country shares)."""
    return [
        {"country": "Indonesia", "pre_samples": 12996, "pre_pct": 33.8, "post_samples": 9155, "post_pct": 36.6},
        {"country": "Malaysia", "pre_samples": 12743, "pre_pct": 33.1, "post_samples": 8400, "post_pct": 33.6},
        {"country": "Thailand", "pre_samples": 10169, "pre_pct": 26.4, "post_samples": 5996, "post_pct": 24.0},
        {"country": "Singapore", "pre_samples": 2469, "pre_pct": 6.4, "post_samples": 1388, "post_pct": 5.6},
        {"country": "Brunei", "pre_samples": 104, "pre_pct": 0.3, "post_samples": 61, "post_pct": 0.2},
    ]


def table5_baseline_validation() -> list[dict[str, Any]]:
    """Table 5 — validation on SEABAD test set (n=5000), three-seed means where noted."""
    return [
        {
            "model": "MobileNetV3-Small",
            "params": "1.1M",
            "accuracy": "99.57 ± 0.25%",
            "auc": "0.9985 ± 0.0002",
            "precision": "0.9956 ± 0.0012",
            "recall": "0.9957 ± 0.0008",
            "note": "Primary edge-deployment baseline",
        },
        {
            "model": "EfficientNetB0",
            "params": "4.4M",
            "accuracy": "99.49 ± 0.23%",
            "auc": "0.9991 ± 0.0004",
            "precision": "0.9959 ± 0.0018",
            "recall": "0.9939 ± 0.0051",
            "note": "",
        },
        {
            "model": "VGG16",
            "params": "14.9M",
            "accuracy": "99.61 ± 0.03%",
            "auc": "0.9995 ± 0.0001",
            "precision": "0.9960 ± 0.0014",
            "recall": "0.9963 ± 0.0010",
            "note": "",
        },
        {
            "model": "ResNet50",
            "params": "24.2M",
            "accuracy": "99.73 ± 0.02%",
            "auc": "0.9992 ± 0.0003",
            "precision": "0.9965 ± 0.0013",
            "recall": "0.9980 ± 0.0012",
            "note": "",
        },
        {
            "model": "BirdNET v2.4",
            "params": "6.5M",
            "accuracy": "68.62%",
            "auc": "0.7819",
            "precision": "0.6499",
            "recall": "0.8072",
            "note": "Zero-shot at τ=0.1; −30.95 pp vs MobileNetV3-Small",
        },
    ]


def positive_pipeline_stages() -> list[dict[str, str]]:
    """Figure 1 — six-stage positive-label branch."""
    return [
        {"stage": 1, "name": "metadata_acquisition", "output": "42,307 records"},
        {"stage": 2, "name": "download_to_flac", "output": "38,494 FLAC"},
        {"stage": 3, "name": "acoustic_deduplication", "output": "38,481 FLAC (13 duplicates removed)"},
        {"stage": 4, "name": "segment_extraction", "output": "3 s WAV clips"},
        {"stage": 5, "name": "species_balancing", "output": "25,000 WAV (Gini 0.601→0.519)"},
        {"stage": 6, "name": "quality_assurance", "output": "25,000 WAV (97.8% audit accuracy)"},
    ]


def framework_card(cfg: SeabadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeabadConfig()
    cov = SundalandCoverage()
    return {
        "name": cfg.title,
        "paper": f"arXiv:{cfg.paper_id}",
        "paper_url": cfg.paper_url,
        "authors": "Muhammad Mun'im Ahmad Zabidi, Mohd Yamani Idna Idris, Norisma Idris (Universiti Malaya)",
        "releases": {
            "zenodo": cfg.zenodo_url,
            "curation_code": cfg.curation_repo_url,
            "validation_code": cfg.validation_repo_url,
        },
        "problem": (
            "PAM in tropical SE Asia needs binary bird-activity detection on 3 s / 16 kHz clips for edge ARUs; "
            "temperate-trained BAD models generalize poorly to dense tropical soundscapes."
        ),
        "dataset": {
            "clips": cfg.total_clips,
            "balance": "50% bird-present, 50% bird-absent",
            "species": cfg.unique_species,
            "clip": f"{cfg.clip_duration_s} s mono {cfg.sample_rate_hz} Hz",
        },
        "curation": {
            "positive_branch": "Xeno-Canto → 6 stages (metadata … QA)",
            "negative_branch": "6 parallel extractions (BirdVox, Freefield, Warblr, FSC-22, ESC-50, DataSEC)",
            "dedup": f"FAISS top-{cfg.faiss_top_k_neighbors} on 256-D mel μ/σ embeddings",
            "balancing": "MiniBatch K-Means (5 clusters/species) + salience-ranked backfill",
            "gini_reduction_pct": round(gini_reduction_pct(cfg.gini_pre_balancing, cfg.gini_post_balancing), 1),
        },
        "coverage": {
            "countries": list(cov.countries),
            "xeno_canto_records": cov.records_after_unknown_removed,
        },
        "limitations": LIMITATIONS,
    }


def headline_results(cfg: SeabadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeabadConfig()
    return {
        "mobilenetv3_accuracy_pct": 99.57,
        "mobilenetv3_auc": 0.9985,
        "birdnet_zero_shot_accuracy_pct": 68.62,
        "domain_shift_accuracy_gap_pp": 30.95,
        "audit_label_accuracy_pct": cfg.audit_accuracy_pct,
        "gini_reduction_pct": round(gini_reduction_pct(cfg.gini_pre_balancing, cfg.gini_post_balancing), 1),
    }


def evaluation_demo(cfg: SeabadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeabadConfig()
    return {"paper": f"arXiv:{cfg.paper_id}", "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle(cfg: SeabadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeabadConfig()
    return {
        "framework": framework_card(cfg),
        "table1_dataset_comparison": table1_dataset_comparison(),
        "table2_negative_sources": table2_negative_sources(),
        "table3_statistics": table3_dataset_statistics(),
        "table4_geography": table4_geographic_distribution(),
        "table5_baselines": table5_baseline_validation(),
        "positive_pipeline": positive_pipeline_stages(),
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }
