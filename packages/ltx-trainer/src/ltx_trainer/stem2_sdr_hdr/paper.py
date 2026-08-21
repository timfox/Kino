"""StEM2 SDR→HDR framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stem2_sdr_hdr._bootstrap import ensure_repo_root

ensure_repo_root()

from gopex_datasets.stem2.constants import (
    DCPOMATIC_HDR_URL,
    EXR_CLOSER_RATIO,
    HDR_PEAK_NITS,
    MEAN_R2,
    PAPER_TITLE,
    SDR_GAMMA,
    SDR_PEAK_NITS,
    STEM2_URL,
    TOTAL_FRAMES,
)

from ltx_trainer.stem2_sdr_hdr.benchmarks import GLOBAL_STATS, PAPER_ARXIV, benchmarks_bundle
from ltx_trainer.stem2_sdr_hdr.pipeline import analyze_synthetic_scene, cinema_mapping_report


def framework_card() -> dict[str, Any]:
    return {
        "name": "StEM2 SDR→HDR",
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "title": PAPER_TITLE,
        "dataset": {
            "name": "ASC StEM2",
            "url": STEM2_URL,
            "total_frames": TOTAL_FRAMES,
            "implementation": DCPOMATIC_HDR_URL,
        },
        "mastering": {
            "sdr": f"DCI-P3 / Gamma {SDR_GAMMA} / {SDR_PEAK_NITS} cd/m²",
            "hdr": f"DCI-P3 / PQ / {HDR_PEAK_NITS} cd/m²",
            "exr": "ACES AP0 linear",
        },
        "key_findings": {
            "mean_isotonic_r2": MEAN_R2,
            "exr_closer_ratio": EXR_CLOSER_RATIO,
            "chroma_correlation": 0.9853,
            "hue_stability_deg": 2.38,
        },
        "global_stats": GLOBAL_STATS,
    }


def evaluation_demo(*, seed: int = 7) -> dict[str, Any]:
    desert = analyze_synthetic_scene("desert", seed=seed)
    cave = analyze_synthetic_scene("cave", seed=seed + 1)
    return {
        "package": "stem2_sdr_hdr",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "framework": framework_card(),
        "benchmarks": benchmarks_bundle(),
        "report": cinema_mapping_report(),
        "synthetic": {"desert": desert, "cave": cave},
    }
