"""Modulo spike HDR paper stub: framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.modulo_spike_hdr.benchmarks import (
    PAPER_ARXIV,
    PAPER_TITLE,
    benchmarks_bundle,
)


def framework_card() -> dict[str, Any]:
    return {
        "name": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "authors": "Zhou, Yang, Zhang, Guo, Yu, Shi, Sato (PKU / BUPT / NII)",
        "problem": (
            "Modulo sensors wrap HDR radiance but prior systems are exposure-coupled, "
            "grayscale, slow, and rely on iterative unwrapping."
        ),
        "method": {
            "formulation": "Decouple representation (dense Uk / spikes) from query (sliding modulo windows)",
            "algorithm": "Iteration-free two-stage: diffusion HDR prior + LAR-physics LMA/CCP refine",
            "hardware": "Spike camera front-end; register-array modulo back-end; 20→6 Gbps",
        },
        "results": {
            "synthetic_psnr_l": 39.17,
            "unwrap_time_per_frame_s": 0.27,
            "capture_fps": 1000,
        },
        "reference_metrics": benchmarks_bundle(),
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.modulo_spike_hdr.mock import evaluation_smoke

    return evaluation_smoke()
