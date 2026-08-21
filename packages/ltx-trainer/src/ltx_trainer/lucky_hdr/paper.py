"""LuckyHDR framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lucky_hdr.benchmarks import TABLE1_SYNTHETIC_3FRAME, benchmarks_bundle
from ltx_trainer.lucky_hdr.capture_ae import bracket_evs
from ltx_trainer.lucky_hdr.layout import LIMITATIONS
from ltx_trainer.lucky_hdr.model import LuckyHdr, LuckyHdrConfig
from ltx_trainer.lucky_hdr.pipeline import pipeline_merge_smoke


def framework_card() -> dict[str, Any]:
    ref = TABLE1_SYNTHETIC_3FRAME["lucky_hdr"]
    return {
        "name": "LuckyHDR",
        "arxiv": "arXiv:2604.19976",
        "design_constraint": "Convex combination merge — no direct HDR regression hallucination",
        "stages": ["shift (φ coarse+fine flow)", "merge (ψ softmax weights)"],
        "reference_metrics": {
            "table1": ref,
            "capture_defaults": {"bracket_frames": 5, "ev_span": 2.0},
        },
        "bracket_evs_default": bracket_evs(n=5, span=2.0),
        "limitations": list(LIMITATIONS),
    }


def pipeline_demo(*, seed: int = 0) -> dict[str, Any]:
    return pipeline_merge_smoke(seed=seed)


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    from ltx_trainer.lucky_hdr.mock import evaluation_smoke

    return {"package": "lucky_hdr", **evaluation_smoke(seed=seed)}


def benchmarks_bundle_export() -> dict[str, Any]:
    return benchmarks_bundle()
