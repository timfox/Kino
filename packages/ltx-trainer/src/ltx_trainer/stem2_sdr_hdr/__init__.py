"""StEM2 cinema SDR→HDR structural analysis stub (Zhang & Chen, arXiv:2604.06276)."""

from ltx_trainer.stem2_sdr_hdr.benchmarks import GLOBAL_STATS, benchmarks_bundle
from ltx_trainer.stem2_sdr_hdr.paper import evaluation_demo, framework_card
from ltx_trainer.stem2_sdr_hdr.pipeline import (
    analyze_custom_triplet,
    analyze_synthetic_scene,
    cinema_mapping_report,
    isotonic_readiness_from_triplet,
)

__all__ = [
    "GLOBAL_STATS",
    "analyze_custom_triplet",
    "analyze_synthetic_scene",
    "benchmarks_bundle",
    "cinema_mapping_report",
    "evaluation_demo",
    "framework_card",
    "isotonic_readiness_from_triplet",
]
