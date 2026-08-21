"""Paper table excerpts for FlatSounds (arXiv:2605.30339)."""

from __future__ import annotations

from typing import Any

# Table 1 — per-metric confidence (captioned, excerpt)
TABLE1_PER_METRIC = {
    "MMAudio-Phys (w/ Caption)": {
        "attack_time": 0.290,
        "decay_rate": 0.310,
        "f0": 0.334,
        "spectral_centroid": 0.368,
        "spectral_flux": 0.321,
        "spectral_rolloff": 0.395,
        "temporal_modulation": 0.237,
        "rt60": 0.310,
        "drr": 0.189,
        "avg": 0.306,
    },
    "Hunyuan-V2A (w/ Caption)": {
        "attack_time": 0.267,
        "decay_rate": 0.320,
        "f0": 0.326,
        "spectral_centroid": 0.332,
        "spectral_flux": 0.403,
        "spectral_rolloff": 0.305,
        "temporal_modulation": 0.283,
        "rt60": 0.247,
        "drr": 0.262,
        "avg": 0.305,
    },
}

# Table 2 — overall FlatSounds-Physics
TABLE2_OVERALL = {
    "MMAudio-Phys (w/ Caption)": {"confidence": 0.306, "hit_coverage": 82.65, "perfect_align": 59.82, "clap": 0.630},
    "Hunyuan-V2A (w/ Caption)": {"confidence": 0.305, "hit_coverage": 90.21, "perfect_align": 69.31, "clap": 0.633},
    "Hunyuan-V2A (w/o Caption)": {"confidence": 0.296, "hit_coverage": 91.50, "perfect_align": 70.50, "clap": 0.593},
    "MMAudio (w/ Caption)": {"confidence": 0.226, "hit_coverage": 75.02, "perfect_align": 52.03, "clap": 0.642},
    "FoleyCrafter (w/ Caption)": {"confidence": 0.205, "hit_coverage": 66.52, "perfect_align": 44.70, "clap": 0.573},
}

# Table 4 — FlatSounds-Single alignment
TABLE4_ALIGNMENT = {
    "Ground Truth": {"hit_coverage": 97.12, "timing_error_ms": 17.25},
    "Hunyuan-V2A (w/o Caption)": {"hit_coverage": 68.55, "timing_error_ms": 44.34},
    "Hunyuan-V2A (w/ Caption)": {"hit_coverage": 65.21, "timing_error_ms": 44.76},
    "MMAudio (w/o Caption)": {"hit_coverage": 75.81, "timing_error_ms": 56.20},
}

# Table 6 — Spearman vs ELO
TABLE6_SPEARMAN = {
    "confidence": 0.9,
    "hit_coverage": 0.9,
    "perfect_align": 0.9,
    "fad_passt": 0.7,
    "desync": 0.7,
    "clap": 0.2,
}

# Table 7 — ELO human study (captioned)
TABLE7_ELO = {
    "Hunyuan-V2A (w/ Caption)": 1556,
    "MMAudio-Phys (w/ Caption)": 1550,
    "ThinkSound (w/ Caption)": 1509,
    "MMAudio (w/ Caption)": 1447,
    "FoleyCrafter (w/ Caption)": 1438,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_per_metric": TABLE1_PER_METRIC,
        "table2_overall": TABLE2_OVERALL,
        "table4_alignment": TABLE4_ALIGNMENT,
        "table6_spearman": TABLE6_SPEARMAN,
        "table7_elo": TABLE7_ELO,
        "caption_hurts_timing": TABLE4_ALIGNMENT["Hunyuan-V2A (w/ Caption)"]["hit_coverage"]
        < TABLE4_ALIGNMENT["Hunyuan-V2A (w/o Caption)"]["hit_coverage"],
        "confidence_beats_desync_spearman": TABLE6_SPEARMAN["confidence"]
        > TABLE6_SPEARMAN["desync"],
        "dataset_clips": 185,
        "physics_test_cases": 268,
    }
