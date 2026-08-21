"""SDR–HDR training corpus and Luma-Eval benchmark (Sec. 5)."""

from __future__ import annotations

TRAINING_SOURCES = {
    "hidrovqa": {"videos": 411, "type": "PGC", "paper": "Saini et al. 2024"},
    "chug": {"videos": 428, "type": "UGC", "paper": "Saini et al. 2025a"},
    "live_tmhdr": {"videos": 40, "type": "PGC", "expert_sdr": True, "paper": "Venkataramanan & Bovik 2024"},
}

TMO_OPERATORS = [
    "Reinhard",
    "BT2446a",
    "BT2446c+GM",
    "HC+GM",
    "2390EETF+GM",
    "YouTube-LogC",
    "OCIOv2",
    "Expert Graded SDR",
]

CRF_LEVELS = (23, 31, 39)

LUMA_EVAL = {
    "videos": 20,
    "sources": ["LIVE-TMHDR (10 held-out)", "CHUG (10 held-out)"],
    "sdr_variants_per_video": 8,
    "crf_levels": list(CRF_LEVELS),
    "encoding": "PQ BT.2020 10-bit",
    "peak_nits": 1000,
}

DATASET_STATS = {
    "total_sdr_hdr_pairs": 318_000,
    "expert_tone_mapped_pairs": 54_000,
    "tmo_count": 8,
    "pgc_ugc_ratio": "1:1",
}


def dataset_summary() -> dict[str, object]:
    return {
        "training_sources": TRAINING_SOURCES,
        "tmo_operators": TMO_OPERATORS,
        "crf_levels": list(CRF_LEVELS),
        "luma_eval": LUMA_EVAL,
        "stats": DATASET_STATS,
    }
