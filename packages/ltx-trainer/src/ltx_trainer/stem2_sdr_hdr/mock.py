"""Runnable evaluation smoke for StEM2 SDR→HDR (arXiv:2604.06276)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stem2_sdr_hdr.benchmarks import GLOBAL_STATS, PAPER_ARXIV, benchmarks_bundle
from ltx_trainer.stem2_sdr_hdr.pipeline import analyze_synthetic_scene, isotonic_readiness_from_triplet


def evaluation_smoke() -> dict[str, Any]:
    desert = analyze_synthetic_scene("desert", seed=11, height=48, width=64)
    cave = analyze_synthetic_scene("cave", seed=13, height=48, width=64)
    readiness = isotonic_readiness_from_triplet("desert", seed=11)
    out: dict[str, Any] = {
        "package": "stem2_sdr_hdr",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_mean_r2": GLOBAL_STATS["mean_isotonic_r2"],
        "ref_exr_closer_ratio": GLOBAL_STATS["exr_closer_ratio"],
        "synthetic_desert_r2": desert["r2"],
        "synthetic_cave_residual_type": cave["residual_type"],
        "synthetic_desert_exr_closer": desert["exr_closer_ratio"],
        "isotonic_readiness_proxy": round(readiness, 4),
    }
    try:
        import torch  # noqa: F401

        out["torch"] = True
        out["gradient_rho_desert"] = round(float(desert["gradient_rho"]), 4)
    except ImportError:
        out["torch"] = False
    return out
